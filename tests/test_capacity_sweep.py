"""Tests for the P1 capacity-sweep harness (src/capacity_sweep.py).

Read implementation/02_testing_philosophy.md before extending this file: a
test that only passes when the code works is worth very little. Every test
here is designed to FAIL if the specific bug it targets is present -- see the
"deliberate breakage" verification log in the task report for evidence that
each one actually does.

Test-to-requirement map (the P1 brief's four blocking conditions):

* Condition 1 (exposures, mistake M4): test_training_steps_for_exposures_*,
  test_exposures_propagate_into_training_steps.
* Condition 2 (precision): test_run_sweep_point_records_dtype_exposures_seed
  (fp32 recorded in every result; no dtype parameter exists anywhere in this
  file to emit a ternary/int4 number).
* Condition 3 (saturation, S4): test_detect_saturation_false_on_straight_line,
  test_detect_saturation_true_on_saturating_curve,
  test_detect_saturation_rejects_a_curve_at_91_percent_of_its_true_asymptote,
  test_bits_per_parameter_at_plateau_raises_when_not_saturated,
  test_bits_per_substrate_byte_at_plateau_raises_when_not_saturated,
  test_detect_saturation_agrees_with_lead_independent_detector.
* Condition 4 (two evaluations): test_tier_a_and_tier_b_draw_from_disjoint_template_pools,
  test_run_sweep_point_tier_b_matches_an_independently_computed_held_out_reference
  (the wiring-level regression test), test_tier_b_on_training_phrasings_collapses_the_gap
  (the observable-consequence test).

End-to-end (task's required test 6): test_tiny_end_to_end_sweep_is_monotone_and_fast.
Determinism (required test 4): test_run_sweep_point_deterministic_at_fixed_seed.
Entropy bound (required test 5): test_bits_recovered_never_exceeds_entropy_ceiling
(unit-level) and test_tiny_end_to_end_sweep_is_monotone_and_fast (on real sweep output).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from src.capacity_sweep import (
    DEFAULT_BLOCK_SIZE,
    MAX_REMAINING_FRACTION,
    MIN_EXPOSURES,
    TRAINING_DTYPE,
    SweepPointResult,
    TierResult,
    UnsaturatedCurveError,
    _build_training_corpus,
    _derive_template_seeds,
    _evaluate_tier,
    _train_family_prompt_answer,
    bits_per_parameter_at_plateau,
    bits_per_substrate_byte_at_plateau,
    build_report,
    detect_saturation,
    run_self_checks,
    run_sweep,
    run_sweep_point,
    substrate_bytes,
    tier_a_prompts,
    tier_b_prompts,
    training_steps_for_exposures,
)
from src.data.generate import generate_people
from src.data.schema import ATTRIBUTES, dataset_entropy_bits
from src.data.tokenizer import PAD, build_tokenizer
from src.models.config import ModelConfig
from src.train import train_loop
from tests.lead.lead_plateau_check import detect_plateau as lead_detect_plateau

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CONTAINER_D_MODEL = 48
_CONTAINER_N_LAYER = 2
_CONTAINER_N_HEAD = 2


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


@pytest.fixture(scope="module")
def small_config(tokenizer):
    return ModelConfig(
        n_layer=_CONTAINER_N_LAYER, n_head=_CONTAINER_N_HEAD, d_model=_CONTAINER_D_MODEL,
        vocab_size=tokenizer.vocab_size, block_size=DEFAULT_BLOCK_SIZE, tie_embeddings=True,
    )


@pytest.fixture(scope="module")
def pad_id(tokenizer):
    return tokenizer.encode(PAD)[0]


# ===========================================================================
# 1. Plateau / saturation detector -- synthetic curves only, never model output.
# ===========================================================================


def test_detect_saturation_false_on_straight_line():
    """A hand-built straight line: constant marginal gain everywhere -> never saturated.

    Gains that never decay have no finite geometric asymptote to extrapolate
    -- decay_ratio == 1.0, remaining_fraction == inf.
    """
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0]
    slope = 50.97  # arbitrary constant; the value does not matter, only that it is constant
    bits_values = [slope * n for n in n_values]

    verdict = detect_saturation(n_values, bits_values)

    print(f"[straight-line] decay={verdict.decay_ratio!r} remaining={verdict.remaining_fraction!r}")
    assert verdict.is_saturated is False
    assert verdict.decay_ratio == pytest.approx(1.0, abs=1e-9)
    assert math.isinf(verdict.remaining_fraction)


def test_detect_saturation_true_on_saturating_curve():
    """A hand-built saturating exponential, reaching 99.97% of its asymptote -> saturated."""
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0, 8_000.0]
    ceiling = 1_000_000.0
    rate = 1e-3  # 1 - exp(-rate * 8000) = 99.97% of ceiling reached by the last point
    bits_values = [ceiling * (1.0 - math.exp(-rate * n)) for n in n_values]

    verdict = detect_saturation(n_values, bits_values)

    print(f"[saturating] decay={verdict.decay_ratio!r} remaining={verdict.remaining_fraction!r}")
    assert verdict.is_saturated is True
    assert verdict.remaining_fraction < MAX_REMAINING_FRACTION


def test_detect_saturation_barely_bending_curve_is_not_saturated():
    """A curve that has reached only 33% of its true asymptote must NOT be called saturated,
    even though it is visibly curving (this is the case a "just look at the shape"
    intuition gets wrong -- the extrapolated remainder is over 2x the measured value)."""
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0, 8_000.0]
    ceiling = 1_000_000.0
    rate = 5e-5  # 1 - exp(-rate * 8000) = 33% of ceiling
    bits_values = [ceiling * (1.0 - math.exp(-rate * n)) for n in n_values]

    verdict = detect_saturation(n_values, bits_values)

    assert verdict.is_saturated is False


def test_detect_saturation_rejects_a_curve_at_91_percent_of_its_true_asymptote():
    """A curve at 90.9% of its true asymptote -- the case tests/lead/lead_plateau_check.py's
    own docstring cites as why a bare ratio-of-marginal-gains threshold is the wrong
    criterion: it has no stated relationship to the actual capacity error. ~9.1% of the
    value is still missing here -- over this file's 5% tolerance -- so it must be
    reported as NOT saturated, even though the curve is visibly bending.

    This file's detect_saturation was not adopted by copying that lesson uncritically:
    this exact curve was checked, before the rewrite, against this file's own
    now-superseded first draft (a ratio of final to steepest-earlier marginal gain,
    threshold 15%) and confirmed correct there too (16.5% > 15%, narrowly and
    coincidentally). The extrapolation criterion below is kept because it bounds the
    quantity that actually matters -- the capacity error -- not because the ratio
    criterion was shown to fail on this specific curve.
    """
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0, 8_000.0]
    ceiling = 1_000_000.0
    rate = 3e-4  # 1 - exp(-rate * 8000) = 90.9% of ceiling
    bits_values = [ceiling * (1.0 - math.exp(-rate * n)) for n in n_values]

    verdict = detect_saturation(n_values, bits_values)

    print(f"[90.9%-of-ceiling] decay={verdict.decay_ratio!r} remaining={verdict.remaining_fraction!r}")
    assert verdict.is_saturated is False
    assert verdict.remaining_fraction == pytest.approx(0.0998, abs=2e-3)


def test_detect_saturation_agrees_with_lead_independent_detector():
    """Cross-check against tests/lead/lead_plateau_check.py's independently-implemented detector.

    Two independent implementations agreeing is real evidence
    (implementation/02_testing_philosophy.md section 1; mistake M9). Both
    files extrapolate the geometric decay of marginal gains and bound the
    remaining fraction -- the same underlying idea (this file adopted it
    after the lead's own file demonstrated a bare ratio threshold was wrong;
    see MAX_REMAINING_FRACTION's provenance comment) -- but estimate the
    decay differently: the lead's detector reads it off the first and last
    gain only; this file fits a least-squares regression through the log of
    every gain (_log_linear_decay_ratio's docstring). Genuinely different
    arithmetic, checked here on the lead's own battery curves.
    """
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0, 8_000.0]

    straight = [100.0 * n for n in n_values]
    at_ceiling = [1_000_000.0 * (1.0 - math.exp(-1e-3 * n)) for n in n_values]
    partial = [1_000_000.0 * (1.0 - math.exp(-3e-4 * n)) for n in n_values]

    for label, curve, expected in (
        ("straight line", straight, False),
        ("99.97% of ceiling", at_ceiling, True),
        ("90.9% of ceiling", partial, False),
    ):
        mine = detect_saturation(n_values, curve)
        lead = lead_detect_plateau(n_values, curve)
        print(f"[cross-check {label}] mine={mine.is_saturated} lead={lead.is_saturated} want={expected}")
        assert mine.is_saturated == lead.is_saturated == expected


def test_detect_saturation_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        detect_saturation([1, 2, 3, 4], [1, 2, 3])


def test_detect_saturation_raises_on_too_few_points():
    with pytest.raises(ValueError, match="need >="):
        detect_saturation([1, 2, 3], [1, 2, 3])


def test_detect_saturation_raises_on_non_increasing_n():
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_saturation([1, 2, 2, 4], [1, 2, 3, 4])


def test_detect_saturation_raises_on_decreasing_bits():
    with pytest.raises(ValueError, match="non-decreasing"):
        detect_saturation([1, 2, 3, 4], [10, 20, 15, 25])


def test_detect_saturation_raises_when_curve_never_started_rising():
    with pytest.raises(ValueError, match="first-interval gain must be positive"):
        detect_saturation([1, 2, 3, 4], [5, 5, 5, 5])


def test_detect_saturation_raises_on_nonpositive_final_value():
    with pytest.raises(ValueError, match="final bits_values must be positive"):
        detect_saturation([1, 2, 3, 4], [1.0, 2.0, 3.0, 0.0])


# ===========================================================================
# 2. bits_per_* MUST RAISE when the curve is not saturated.
# ===========================================================================


def test_bits_per_parameter_at_plateau_raises_when_not_saturated():
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0]
    bits_values = [50.0 * n for n in n_values]  # straight line: never saturated

    with pytest.raises(UnsaturatedCurveError, match="not saturated"):
        bits_per_parameter_at_plateau(n_values, bits_values, parameter_count=1_000)


def test_bits_per_substrate_byte_at_plateau_raises_when_not_saturated():
    n_values = [1_000.0, 2_000.0, 3_000.0, 4_000.0]
    bits_values = [50.0 * n for n in n_values]

    with pytest.raises(UnsaturatedCurveError, match="not saturated"):
        bits_per_substrate_byte_at_plateau(n_values, bits_values, substrate_bytes_value=4_000)


def test_bits_per_parameter_at_plateau_succeeds_when_saturated():
    # rate=1e-3 over this N range reaches 99.97% of its asymptote (well under
    # the 5% remaining-fraction tolerance) -- clearly saturated, not a
    # borderline case.
    n_values = [1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0]
    bits_values = [100_000.0 * (1.0 - math.exp(-1e-3 * n)) for n in n_values]

    result = bits_per_parameter_at_plateau(n_values, bits_values, parameter_count=50_000)

    assert result == pytest.approx(bits_values[-1] / 50_000)


def test_bits_per_substrate_byte_at_plateau_succeeds_when_saturated():
    n_values = [1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0]
    bits_values = [100_000.0 * (1.0 - math.exp(-1e-3 * n)) for n in n_values]

    result = bits_per_substrate_byte_at_plateau(n_values, bits_values, substrate_bytes_value=200_000)

    assert result == pytest.approx(bits_values[-1] / 200_000)


def test_bits_per_parameter_at_plateau_rejects_nonpositive_parameter_count():
    n_values = [1_000.0, 2_000.0, 4_000.0, 8_000.0]
    bits_values = [100_000.0 * (1.0 - math.exp(-1e-3 * n)) for n in n_values]
    with pytest.raises(ValueError):
        bits_per_parameter_at_plateau(n_values, bits_values, parameter_count=0)


# ===========================================================================
# 3. Exposures propagate into results and change the step count.
# ===========================================================================


def test_training_steps_for_exposures_basic():
    assert training_steps_for_exposures(exposures=100, n_train_examples=40, batch_size=8) == 500


def test_training_steps_for_exposures_rounds_up():
    # 30 * 16 / 8 = 60 exactly; nudge to a non-multiple to exercise ceil().
    assert training_steps_for_exposures(exposures=31, n_train_examples=16, batch_size=8) == 62
    assert training_steps_for_exposures(exposures=1, n_train_examples=1, batch_size=8) == 1


def test_exposures_propagate_into_training_steps():
    """Doubling exposures must double the step count -- the M4 wiring itself."""
    low = training_steps_for_exposures(exposures=50, n_train_examples=64, batch_size=8)
    high = training_steps_for_exposures(exposures=100, n_train_examples=64, batch_size=8)

    print(f"[exposures->steps] exposures=50 -> {low} steps; exposures=100 -> {high} steps")
    assert high == 2 * low
    assert low != high


def test_training_steps_for_exposures_rejects_below_minimum():
    with pytest.raises(ValueError, match="MIN_EXPOSURES" if False else "exposures must be >="):
        training_steps_for_exposures(exposures=0, n_train_examples=10, batch_size=8)
    assert MIN_EXPOSURES == 1  # the documented floor this test targets


def test_training_steps_for_exposures_rejects_nonpositive_examples_or_batch():
    with pytest.raises(ValueError):
        training_steps_for_exposures(exposures=10, n_train_examples=0, batch_size=8)
    with pytest.raises(ValueError):
        training_steps_for_exposures(exposures=10, n_train_examples=10, batch_size=0)


# ===========================================================================
# substrate_bytes accounting
# ===========================================================================


def test_substrate_bytes_fp32_default():
    assert substrate_bytes(1_000) == 4_000  # fp32 = 4 bytes/param, no extra state


def test_substrate_bytes_includes_persistent_state_and_index():
    total = substrate_bytes(1_000, persistent_state_bytes=500, index_bytes=250)
    assert total == 4_000 + 500 + 250


def test_substrate_bytes_rejects_nonpositive_parameter_count():
    with pytest.raises(ValueError):
        substrate_bytes(0)
    with pytest.raises(ValueError):
        substrate_bytes(-5)


def test_substrate_bytes_rejects_negative_state_or_index():
    with pytest.raises(ValueError):
        substrate_bytes(1_000, persistent_state_bytes=-1)
    with pytest.raises(ValueError):
        substrate_bytes(1_000, index_bytes=-1)


# ===========================================================================
# 4 & 5 (blocking conditions). Tier A / Tier B construction, and the P1 brief's
# open research question: memorisation vs extractable-knowledge gap.
# ===========================================================================


def test_train_family_prompt_answer_splits_at_the_value():
    sentence = "Kelvan Vushiel-1234 grew up in Tarnvale."
    prompt, answer = _train_family_prompt_answer(sentence, "Tarnvale")
    assert answer == "Tarnvale"
    assert prompt == "Kelvan Vushiel-1234 grew up in "
    assert prompt + answer + "." == sentence


def test_train_family_prompt_answer_raises_if_value_absent():
    with pytest.raises(ValueError, match="occur exactly once"):
        _train_family_prompt_answer("Kelvan Vushiel-1234 grew up in Tarnvale.", "NotPresent")


def test_train_family_prompt_answer_raises_if_value_not_unique():
    with pytest.raises(ValueError, match="occur exactly once"):
        _train_family_prompt_answer("Tarn Tarnvale grew up in Tarnvale.", "Tarnvale")


def test_tier_a_and_tier_b_draw_from_disjoint_template_pools():
    """Tier A must use TRAIN-family (declarative) phrasing, Tier B held-out PROBE
    (interrogative) phrasing -- structurally distinguishable, never the same text.

    Every src.data.generate PROBE template is a question ending in "?"
    (verified directly in that file: implementation/04_golden_dataset.md
    section 6's examples); every TRAIN template is a declarative statement
    that never contains "?". This structural difference is what catches a
    mistaken swap of which template pool feeds which tier -- the exact
    mutation the task's acceptance section asks to demonstrate (see
    test_tier_b_on_training_phrasings_collapses_the_gap for the same
    mutation's effect on the measured A/B gap).
    """
    person = next(iter(generate_people(1, seed=99)))
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)

    a_prompts = tier_a_prompts(person, rng_a)
    b_prompts = tier_b_prompts(person, rng_b)

    for attribute in ATTRIBUTES:
        a_prompt, a_answer = a_prompts[attribute]
        b_prompt, b_answer = b_prompts[attribute]

        assert a_answer == person.attributes[attribute]
        assert b_answer == person.attributes[attribute]
        assert "?" not in a_prompt, f"Tier A ({attribute}) must be declarative, got {a_prompt!r}"
        assert b_prompt.rstrip().endswith("?"), f"Tier B ({attribute}) must be interrogative, got {b_prompt!r}"
        assert a_prompt != b_prompt


def test_run_sweep_point_tier_b_matches_an_independently_computed_held_out_reference(
    small_config, tokenizer, pad_id
):
    """run_sweep_point's Tier B must be wired to tier_b_prompts, not tier_a_prompts.

    This is the regression test for the exact mutation the task's acceptance
    section asks to demonstrate ("evaluate Tier B on training phrasings").
    Earlier drafts of this test trained a model once via run_sweep_point and
    a second time by hand, then compared run_sweep_point's own tier_b against
    a manually-recomputed "fake tier_b" -- but that comparison is vacuous
    against exactly this mutation: if run_sweep_point's internal wiring is
    broken, its OWN point.tier_b is already the broken value, so comparing it
    to a second, independently-broken computation trivially "agrees" (both
    sides are wrong the same way). Confirmed empirically while building this
    file: that version of the test still passed under the mutation.

    The fix is to build two INDEPENDENT reference evaluations -- one using
    the correct tier_b_prompts, one using tier_a_prompts standing in for it
    -- on a model trained separately from run_sweep_point (same seed, so
    bit-identical weights per the determinism test above), and assert
    run_sweep_point's point.tier_b matches the CORRECT reference, not the
    broken one. Only this shape can fail when the wiring is wrong.
    """
    seed = 7
    n_people = 16
    exposures = 300
    batch_size = 8
    train_seed, tier_a_seed, tier_b_seed = _derive_template_seeds(seed)

    point = run_sweep_point(
        n_people=n_people, seed=seed, exposures=exposures, batch_size=batch_size, lr=3e-4,
        weight_decay=0.0, config=small_config, tokenizer=tokenizer, pad_id=pad_id,
        train_template_seed=train_seed, tier_a_template_seed=tier_a_seed,
        tier_b_template_seed=tier_b_seed, num_threads=4,
    )

    # Independently reconstruct the same trained model (determinism test
    # above already establishes this reproduces bit-identical weights).
    people = list(generate_people(n_people, seed))
    train_rng = np.random.default_rng(train_seed)
    data = _build_training_corpus(people, tokenizer, small_config.block_size, train_rng, pad_id)
    steps = training_steps_for_exposures(exposures, n_train_examples=n_people, batch_size=batch_size)
    trained = train_loop(
        config=small_config, seed=seed, steps=steps, batch_size=batch_size, lr=3e-4,
        data=data, num_threads=4, weight_decay=0.0,
    )
    model = trained.model
    model.eval()

    correct_reference = _evaluate_tier(
        model, tokenizer, people, tier_b_prompts, np.random.default_rng(tier_b_seed)
    )
    broken_reference = _evaluate_tier(
        model, tokenizer, people, tier_a_prompts, np.random.default_rng(tier_b_seed)
    )

    print(f"[tier-b wiring] point.tier_b            = {point.tier_b.bits_recovered:.2f} bits")
    print(f"[tier-b wiring] correct (held-out) ref   = {correct_reference.bits_recovered:.2f} bits")
    print(f"[tier-b wiring] broken (train-family) ref= {broken_reference.bits_recovered:.2f} bits")

    assert point.tier_b.bits_recovered == correct_reference.bits_recovered
    # The two references must actually differ, or this test could not have
    # distinguished correct from broken wiring in the first place.
    assert correct_reference.bits_recovered != broken_reference.bits_recovered


def test_tier_b_on_training_phrasings_collapses_the_gap(small_config, tokenizer, pad_id):
    """Deliberately evaluate 'Tier B' using TRAIN-family phrasing (the mutation the
    acceptance section asks to demonstrate) and confirm the A/B gap collapses --
    a real, observable consequence of the tier-separation breaking, reportable
    even without inspecting run_sweep_point's internal wiring directly (that
    is what the sibling test above does).
    """
    seed = 7
    n_people = 16
    exposures = 300
    batch_size = 8
    train_seed, tier_a_seed, tier_b_seed = _derive_template_seeds(seed)

    point = run_sweep_point(
        n_people=n_people, seed=seed, exposures=exposures, batch_size=batch_size, lr=3e-4,
        weight_decay=0.0, config=small_config, tokenizer=tokenizer, pad_id=pad_id,
        train_template_seed=train_seed, tier_a_template_seed=tier_a_seed,
        tier_b_template_seed=tier_b_seed, num_threads=4,
    )
    real_gap = point.tier_a.bits_recovered - point.tier_b.bits_recovered

    people = list(generate_people(n_people, seed))
    train_rng = np.random.default_rng(train_seed)
    data = _build_training_corpus(people, tokenizer, small_config.block_size, train_rng, pad_id)
    steps = training_steps_for_exposures(exposures, n_train_examples=n_people, batch_size=batch_size)
    trained = train_loop(
        config=small_config, seed=seed, steps=steps, batch_size=batch_size, lr=3e-4,
        data=data, num_threads=4, weight_decay=0.0,
    )
    model = trained.model
    model.eval()

    fake_tier_b = _evaluate_tier(
        model, tokenizer, people, tier_a_prompts, np.random.default_rng(tier_b_seed)
    )
    broken_gap = point.tier_a.bits_recovered - fake_tier_b.bits_recovered

    print(f"[tier-swap mutation] real gap (A - held-out B) = {real_gap:.2f} bits")
    print(f"[tier-swap mutation] broken gap (A - fake B=TRAIN family) = {broken_gap:.2f} bits")

    # The broken gap must be small (model scores itself similarly on two
    # independent draws from the SAME template family) and must be smaller
    # than the real, held-out-phrasing gap whenever the real gap is
    # meaningfully positive.
    assert abs(broken_gap) < max(1.0, 0.25 * point.tier_a.bits_recovered)
    if real_gap > 5.0:  # only a meaningful real gap can "collapse" relative to itself
        assert abs(broken_gap) < real_gap


# ===========================================================================
# Determinism (required test 4).
# ===========================================================================


def test_run_sweep_point_deterministic_at_fixed_seed(small_config, tokenizer, pad_id):
    seed = 3
    train_seed, tier_a_seed, tier_b_seed = _derive_template_seeds(seed)
    kwargs = dict(
        n_people=16, seed=seed, exposures=20, batch_size=8, lr=3e-4, weight_decay=0.0,
        config=small_config, tokenizer=tokenizer, pad_id=pad_id,
        train_template_seed=train_seed, tier_a_template_seed=tier_a_seed,
        tier_b_template_seed=tier_b_seed, num_threads=4,
    )

    run_1 = run_sweep_point(**kwargs)
    run_2 = run_sweep_point(**kwargs)

    print(f"[determinism] run1 tier_a={run_1.tier_a.bits_recovered!r} run2={run_2.tier_a.bits_recovered!r}")
    assert run_1.tier_a.bits_recovered == run_2.tier_a.bits_recovered  # bit-identical, not approximate
    assert run_1.tier_b.bits_recovered == run_2.tier_b.bits_recovered
    assert run_1.final_train_loss_nats == run_2.final_train_loss_nats
    assert run_1.training_steps == run_2.training_steps
    assert run_1.parameter_count == run_2.parameter_count


def test_run_sweep_point_records_dtype_exposures_seed(small_config, tokenizer, pad_id):
    """Blocking conditions 1 & 2: every result names its exposures, seed, and dtype."""
    train_seed, tier_a_seed, tier_b_seed = _derive_template_seeds(11)
    point = run_sweep_point(
        n_people=16, seed=11, exposures=17, batch_size=8, lr=3e-4, weight_decay=0.0,
        config=small_config, tokenizer=tokenizer, pad_id=pad_id,
        train_template_seed=train_seed, tier_a_template_seed=tier_a_seed,
        tier_b_template_seed=tier_b_seed, num_threads=4,
    )
    assert point.exposures == 17
    assert point.seed == 11
    assert point.dtype == str(TRAINING_DTYPE) == "torch.float32"


# ===========================================================================
# Entropy bound (required test 5): bits_recovered <= entropy ceiling at every N.
# ===========================================================================


def test_bits_recovered_never_exceeds_entropy_ceiling(small_config, tokenizer, pad_id):
    train_seed, tier_a_seed, tier_b_seed = _derive_template_seeds(5)
    for n in (10, 20, 40):
        point = run_sweep_point(
            n_people=n, seed=5, exposures=25, batch_size=10, lr=3e-4, weight_decay=0.0,
            config=small_config, tokenizer=tokenizer, pad_id=pad_id,
            train_template_seed=train_seed, tier_a_template_seed=tier_a_seed,
            tier_b_template_seed=tier_b_seed, num_threads=4,
        )
        expected_ceiling = dataset_entropy_bits(n)
        print(f"[entropy-bound] N={n} tier_a={point.tier_a.bits_recovered:.2f} "
              f"tier_b={point.tier_b.bits_recovered:.2f} ceiling={expected_ceiling:.2f}")
        assert point.tier_a.entropy_ceiling == pytest.approx(expected_ceiling)
        assert point.tier_b.entropy_ceiling == pytest.approx(expected_ceiling)
        assert 0.0 <= point.tier_a.bits_recovered <= point.tier_a.entropy_ceiling
        assert 0.0 <= point.tier_b.bits_recovered <= point.tier_b.entropy_ceiling


# ===========================================================================
# 6. Tiny end-to-end sweep, UNDER 3 MINUTES on 4 CPU cores, monotone curve.
# ===========================================================================


def test_tiny_end_to_end_sweep_is_monotone_and_fast(tmp_path: Path):
    """The full CLI-equivalent pipeline (run_sweep): generate -> train -> evaluate ->
    report, across several N, resumably, producing a monotone non-decreasing curve.

    Parameters below were tuned (see task report) so the SMALLEST N gets
    enough absolute optimizer steps to move meaningfully off the chance-level
    clamp -- at very low absolute step counts the curve can be flat-then-
    jump, which is monotone but a weak demonstration. This exact (seed,
    exposures, sizes) combination is deterministic, so its outcome does not
    depend on being lucky at test time.
    """
    out_path = tmp_path / "tiny_sweep.json"
    started = time.monotonic()

    # exposures=450 was chosen empirically (see task report): at lower
    # exposures (e.g. 150) the smallest N's are undertrained enough that
    # Tier B (the harder, held-out-phrasing tier) noisily dips above and back
    # to its zero clamp instead of cleanly rising -- monotone in the
    # mathematical sense but not a meaningful demonstration. At 450 both
    # tiers rise cleanly. Fixed seed makes this deterministic, not lucky.
    payload = run_sweep(
        sizes=[8, 16, 32, 64], seed=0, exposures=450, batch_size=8, lr=3e-4, weight_decay=0.0,
        n_layer=_CONTAINER_N_LAYER, n_head=_CONTAINER_N_HEAD, d_model=_CONTAINER_D_MODEL,
        block_size=DEFAULT_BLOCK_SIZE, num_threads=4, out_path=out_path, resume=False,
    )

    elapsed = time.monotonic() - started
    print(f"[end-to-end] elapsed={elapsed:.1f}s (budget: 180s)")
    assert elapsed < 180.0, f"tiny end-to-end sweep took {elapsed:.1f}s, over the 3-minute budget"

    points = sorted(payload["points"], key=lambda p: p["n_people"])
    assert [p["n_people"] for p in points] == [8, 16, 32, 64]

    tier_a_curve = [p["tier_a"]["bits_recovered"] for p in points]
    tier_b_curve = [p["tier_b"]["bits_recovered"] for p in points]
    print(f"[end-to-end] tier_a curve = {tier_a_curve}")
    print(f"[end-to-end] tier_b curve = {tier_b_curve}")

    assert all(b >= a - 1e-9 for a, b in zip(tier_a_curve, tier_a_curve[1:])), (
        f"tier_a bits_recovered is not monotone non-decreasing in N: {tier_a_curve}"
    )
    assert all(b >= a - 1e-9 for a, b in zip(tier_b_curve, tier_b_curve[1:])), (
        f"tier_b bits_recovered is not monotone non-decreasing in N: {tier_b_curve}"
    )

    # Out file exists, is valid JSON, and round-trips.
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["points"] == payload["points"]

    # Entropy bound holds at every N (required test 5, on real sweep output).
    for p in points:
        assert p["tier_a"]["bits_recovered"] <= p["tier_a"]["entropy_ceiling"]
        assert p["tier_b"]["bits_recovered"] <= p["tier_b"]["entropy_ceiling"]

    # Every result names its exposures, seed, and dtype (blocking conditions 1, 2).
    for p in points:
        assert p["exposures"] == 450
        assert p["seed"] == 0
        assert p["dtype"] == "torch.float32"


def test_tiny_sweep_resumes_to_the_same_result_as_an_uninterrupted_run(tmp_path: Path):
    """A sweep killed partway through and resumed with the SAME --sizes must match
    an uninterrupted run bit-for-bit.

    This is the realistic "quota ran out" scenario
    (implementation/01_phases_and_gates.md, "Quota and interruption policy"):
    the human re-issues the identical command plus --resume, not a shorter or
    longer --sizes list. The interruption is simulated with an
    on_point_done callback that raises after two points -- by then,
    _atomic_write_json has already persisted those two points, exactly as a
    real process kill at any moment after a point completes would leave
    behind a valid (if incomplete) file. Also exercises the
    Training/Concurrent edge case: a fresh run without --resume into an
    occupied path must refuse.
    """
    kwargs = dict(
        sizes=[8, 16, 24, 32], seed=1, exposures=20, batch_size=8, lr=3e-4, weight_decay=0.0,
        n_layer=_CONTAINER_N_LAYER, n_head=_CONTAINER_N_HEAD, d_model=_CONTAINER_D_MODEL,
        block_size=DEFAULT_BLOCK_SIZE, num_threads=4,
    )

    straight_path = tmp_path / "straight.json"
    straight = run_sweep(out_path=straight_path, resume=False, **kwargs)

    class _SimulatedKill(RuntimeError):
        """Stands in for a process kill / quota exhaustion mid-sweep."""

    completed: list[int] = []

    def _kill_after_two(point: SweepPointResult) -> None:
        completed.append(point.n_people)
        if len(completed) == 2:
            raise _SimulatedKill("quota exhausted")

    resumed_path = tmp_path / "resumed.json"
    with pytest.raises(_SimulatedKill):
        run_sweep(out_path=resumed_path, resume=False, on_point_done=_kill_after_two, **kwargs)

    partial = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert {p["n_people"] for p in partial["points"]} == {8, 16}

    # A fresh (non-resume) run into the same path must refuse to interleave.
    with pytest.raises(FileExistsError):
        run_sweep(out_path=resumed_path, resume=False, **kwargs)

    resumed = run_sweep(out_path=resumed_path, resume=True, **kwargs)

    straight_by_n = {p["n_people"]: p for p in straight["points"]}
    resumed_by_n = {p["n_people"]: p for p in resumed["points"]}
    assert straight_by_n.keys() == resumed_by_n.keys() == {8, 16, 24, 32}
    for n in straight_by_n:
        assert straight_by_n[n]["tier_a"]["bits_recovered"] == resumed_by_n[n]["tier_a"]["bits_recovered"]
        assert straight_by_n[n]["tier_b"]["bits_recovered"] == resumed_by_n[n]["tier_b"]["bits_recovered"]
        assert straight_by_n[n]["training_steps"] == resumed_by_n[n]["training_steps"]


def test_resume_rejects_mismatched_config(tmp_path: Path):
    out_path = tmp_path / "sweep.json"
    kwargs = dict(
        seed=2, batch_size=8, lr=3e-4, weight_decay=0.0, n_layer=_CONTAINER_N_LAYER,
        n_head=_CONTAINER_N_HEAD, d_model=_CONTAINER_D_MODEL, block_size=DEFAULT_BLOCK_SIZE,
        num_threads=4,
    )
    run_sweep(sizes=[8, 16], exposures=10, out_path=out_path, resume=False, **kwargs)

    with pytest.raises(ValueError, match="cannot resume"):
        run_sweep(sizes=[8, 16], exposures=999, out_path=out_path, resume=True, **kwargs)


def test_resume_without_existing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_sweep(
            sizes=[8, 16], seed=0, exposures=10, batch_size=8, lr=3e-4, weight_decay=0.0,
            n_layer=2, n_head=2, d_model=32, block_size=DEFAULT_BLOCK_SIZE, num_threads=4,
            out_path=tmp_path / "does_not_exist.json", resume=True,
        )


# ===========================================================================
# build_report / _tier_capacity_report edge cases.
# ===========================================================================


def test_build_report_reports_insufficient_points_before_min_sweep_points():
    points = [
        SweepPointResult(
            n_people=n, exposures=10, training_steps=10, seed=0, dtype="torch.float32",
            parameter_count=100, substrate_bytes=400, batch_size=8, lr=3e-4,
            final_train_loss_nats=1.0, wall_clock_seconds=0.1,
            tier_a=_tier(bits), tier_b=_tier(bits / 2),
            gap_a_minus_b_bits=bits / 2,
        )
        for n, bits in ((10, 5.0), (20, 10.0))
    ]
    report = build_report(points)
    assert "insufficient_points" in report["tier_a"]
    assert "insufficient_points" in report["tier_b"]


def _tier(bits_recovered: float) -> TierResult:
    return TierResult(bits_recovered=bits_recovered, entropy_ceiling=1_000.0, per_attribute_bits={})


def test_build_report_does_not_crash_on_a_flat_then_rising_curve():
    """The real edge case hit while building this file: two flat-zero points
    followed by a rising one gives detect_saturation no positive reference
    slope. build_report must report this gracefully, not raise."""
    points = [
        SweepPointResult(
            n_people=n, exposures=10, training_steps=10, seed=0, dtype="torch.float32",
            parameter_count=100, substrate_bytes=400, batch_size=8, lr=3e-4,
            final_train_loss_nats=1.0, wall_clock_seconds=0.1,
            tier_a=_tier(bits), tier_b=_tier(bits), gap_a_minus_b_bits=0.0,
        )
        for n, bits in ((8, 0.0), (16, 0.0), (32, 0.0), (64, 9.0))
    ]
    report = build_report(points)
    assert "cannot_judge_saturation_reason" in report["tier_a"]
    assert report["tier_a"]["bits_per_parameter"] is None


# ===========================================================================
# Self-checks and CLI plumbing.
# ===========================================================================


def test_run_self_checks_passes():
    run_self_checks()  # must not raise


def test_cli_sizes_parsing_rejects_duplicates_and_empty():
    from src.capacity_sweep import _parse_sizes

    assert _parse_sizes("16,8,32") == (8, 16, 32)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_sizes("8,16,8")
    with pytest.raises(ValueError, match="positive"):
        _parse_sizes("8,-1,16")
    with pytest.raises(ValueError):
        _parse_sizes("")


def test_cli_main_writes_report_and_exits_zero(tmp_path: Path):
    from src.capacity_sweep import main

    out_path = tmp_path / "cli_sweep.json"
    exit_code = main([
        "--seed", "0", "--exposures", "10", "--sizes", "8,16,24,32",
        "--batch-size", "8", "--n-layer", "2", "--n-head", "2", "--d-model", "32",
        "--block-size", str(DEFAULT_BLOCK_SIZE), "--num-threads", "4", "--out", str(out_path),
    ])
    assert exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["points"]) == 4
    assert "report" in payload
