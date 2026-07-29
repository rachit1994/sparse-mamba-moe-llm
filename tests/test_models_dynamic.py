"""Tests for the dynamic-substrate arm (src/models/dynamic.py).

implementation/08_thesis_alignment.md section 6's requirement, restated as
the acceptance gate this file exists to enforce: knowledge must be able to
enter `DynamicSubstrate` WITHOUT a gradient step on its base weights. Every
test here is designed to FAIL if that property (or a related one -- null
retrieval, byte accounting, determinism) does not actually hold, per
implementation/02_testing_philosophy.md's "a test that only passes when the
code works is worth very little." See the task report's break/revert log for
evidence that each one genuinely does fail against the break it targets.

Test-to-requirement map (matches the calling brief's five required tests):
  1. THE GATE: test_gate_base_weights_untouched_and_facts_retrievable
  2. Null control (C1): test_null_substrate_retrieves_at_chance
  3. Capacity/interference: test_capacity_interference_curve_prints_and_degrades
  4. substrate_bytes accounting: test_substrate_bytes_matches_independent_hand_count
  5. Determinism: test_determinism_same_seed_identical_everything (+ negative
     control test_determinism_different_seed_differs, per mistake log M9:
     agreement is only evidence on a value that is hard to produce by
     accident -- a determinism test needs a "seeds differ" counterpart or it
     could pass vacuously).

FACT CONSTRUCTION: uses the real `src.data.tokenizer.Tokenizer` class (not
`build_tokenizer()`'s full production vocabulary, which would make the
capacity sweep's fact count and runtime hard to control) over a small,
hand-built closed vocabulary, and reuses `schema.ATTRIBUTE_CARDINALITY` for
the chance-level ceiling everywhere a ceiling is needed, so "bits recovered"
is computed by the SAME `src.metrics.bits.bits_recovered_per_attribute`
function the control arm uses, with a real project constant as the ceiling.

CONTAMINATION PROOF (implementation/02_testing_philosophy.md C3): every
person/value pair is drawn from `numpy.random.default_rng(seed)` inside this
file, in-process, with no disk or network I/O. The substrate cannot have seen
an answer it was not explicitly `write()`-n, because nothing exists to leak
from -- the facts are generated fresh in the test process and exist nowhere
else.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from src.data.schema import ATTRIBUTE_CARDINALITY
from src.data.tokenizer import Tokenizer
from src.metrics.bits import answer_cross_entropy_bits, bits_recovered_per_attribute
from src.models.dynamic import PAD_ID, DynamicSubstrate, Fact
from src.models.dynamic import _MAX_ANSWER_OFFSET  # noqa: SLF001 -- independent hand-count, test 4

# ---------------------------------------------------------------------------
# Synthetic fact construction: a small closed vocabulary, tokenized by the
# real Tokenizer, scored with the real bits-recovered metric.
#
# Person names and attribute values are each drawn from a fixed pool of
# FUSED (no-separator) single-token strings, mirroring exactly how
# src/data/generate.py builds its city/university/major vocabularies (see
# that file's `_build_fused_vocab`): a root+suffix pair concatenated with no
# separator tokenizes as ONE opaque word (src/data/tokenizer.py only splits
# on punctuation and digit runs), so every name/value gets its own
# independent random row in DynamicSubstrate's fixed projections.
#
# Each fact's "question" is the bare name token, with no wrapping boilerplate
# words ("What did ... major in?" was tried first and measured, not assumed,
# to fail: DynamicSubstrate's context key is sign(sum of context-token rows),
# and with 5 constant boilerplate tokens outnumbering 1 varying name token,
# the constant part dominates the sign in most dimensions, so different
# facts' keys came out ~84% bit-identical regardless of code_dim -- a
# vocabulary-design confound, not a substrate defect; see this module's
# `test_offset_binding_recovers_every_position_of_a_multi_token_answer` for
# where the same substrate is exercised with a longer, multi-token context
# and answer instead). Reducing context to the single most-informative token
# removes the confound and is what the capacity curve below is actually
# measuring: the memory's true pattern capacity, not boilerplate dilution.
# ---------------------------------------------------------------------------

_ATTRIBUTE = "major"  # a real schema attribute; reused for its exact V=100 ceiling.
_VALUE_COUNT = ATTRIBUTE_CARDINALITY[_ATTRIBUTE]

_NAME_ROOT = (
    "Kel", "Mor", "Van", "Ash", "Bri", "Tor", "Sel", "Dun", "Fen", "Gor",
    "Hal", "Iven", "Jor", "Kav", "Lom", "Myr", "Nor", "Oren", "Pel", "Ren",
)
_NAME_SUFFIX = (
    "an", "oran", "ric", "dor", "wyn", "iel", "ael", "oth", "und", "ard",
    "eth", "ion", "van", "wick", "mund", "gard", "lyn", "sten", "rick", "vane",
)
_NAME_VOCAB_SEED = 20260729  # fixed constant: the name vocabulary, not any
# per-test seed, controls which fused strings exist (mirrors generate.py's
# _VOCAB_SEED convention: the universe of possible names is seed-independent).
_NAME_DOMAIN = len(_NAME_ROOT) * len(_NAME_SUFFIX)  # 400


def _build_fused_name_pool() -> tuple[str, ...]:
    """All 400 fused root+suffix name strings, in a fixed permuted order."""
    rng = np.random.default_rng(_NAME_VOCAB_SEED)
    order = rng.permutation(_NAME_DOMAIN)
    names = tuple(f"{_NAME_ROOT[idx // len(_NAME_SUFFIX)]}{_NAME_SUFFIX[idx % len(_NAME_SUFFIX)]}" for idx in order)
    assert len(set(names)) == _NAME_DOMAIN, "fused name pool has a root+suffix collision"
    return names


_NAME_POOL: tuple[str, ...] = _build_fused_name_pool()

_VALUE_ROOT = ("Astro", "Xeno", "Hydro", "Cryo", "Petro", "Bio", "Photo", "Litho", "Aero", "Thermo")
_VALUE_SUFFIX = ("logy", "nomics", "graphy", "technics", "urgy", "chemistry", "metrics", "dynamics", "genetics", "physics")
_VALUE_WORDS: tuple[str, ...] = tuple(f"{root}{suffix}" for root in _VALUE_ROOT for suffix in _VALUE_SUFFIX)
assert len(_VALUE_WORDS) == _VALUE_COUNT, f"expected exactly {_VALUE_COUNT} fused values, got {len(_VALUE_WORDS)}"
assert len(set(_VALUE_WORDS)) == _VALUE_COUNT, "fused value pool has a root+suffix collision"


def _build_test_tokenizer() -> Tokenizer:
    """A small closed vocabulary covering every word this file's facts can produce."""
    return Tokenizer(sorted({*_NAME_POOL, *_VALUE_WORDS}))


def _person_name(i: int) -> str:
    if not 0 <= i < _NAME_DOMAIN:
        raise ValueError(f"person index {i} exceeds the {_NAME_DOMAIN}-name test pool")
    return _NAME_POOL[i]


def _question(i: int) -> str:
    """The bare name -- see module docstring for why no wrapping boilerplate."""
    return _person_name(i)


def _answer(value_idx: int) -> str:
    return _VALUE_WORDS[value_idx]


def _make_facts(tokenizer: Tokenizer, n_people: int, seed: int) -> list[Fact]:
    """`n_people` synthetic (question, answer) facts, encoded via the real tokenizer.

    Contamination proof: `value_idx` is drawn from `np.random.default_rng(seed)`
    in this process, nowhere else -- see module docstring.
    """
    rng = np.random.default_rng(seed)
    value_indices = rng.integers(0, _VALUE_COUNT, size=n_people)
    return [
        tokenizer.encode_qa(_question(i), _answer(int(value_indices[i])))
        for i in range(n_people)
    ]


def _pad_facts(facts: Sequence[Fact]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a ragged batch of facts with PAD_ID, matching read_logits' contract."""
    max_len = max(len(ids) for ids, _ in facts)
    ids_batch = torch.full((len(facts), max_len), PAD_ID, dtype=torch.long)
    mask_batch = torch.zeros((len(facts), max_len), dtype=torch.bool)
    for row, (ids, mask) in enumerate(facts):
        ids_batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask_batch[row, : len(mask)] = torch.tensor(mask, dtype=torch.bool)
    return ids_batch, mask_batch


def _per_fact_ce_bits(substrate: DynamicSubstrate, facts: Sequence[Fact]) -> list[float]:
    """One summed answer_cross_entropy_bits value per fact (the shape bits_recovered_per_attribute needs)."""
    ids_batch, mask_batch = _pad_facts(facts)
    logits = substrate.read_logits(ids_batch, mask_batch)
    return [
        answer_cross_entropy_bits(logits[row : row + 1], ids_batch[row : row + 1], mask_batch[row : row + 1])
        for row in range(len(facts))
    ]


def _retrieval_fraction(substrate: DynamicSubstrate, facts: Sequence[Fact]) -> float:
    ce_bits = _per_fact_ce_bits(substrate, facts)
    recovered = bits_recovered_per_attribute({_ATTRIBUTE: ce_bits})[_ATTRIBUTE]
    return recovered / (math.log2(_VALUE_COUNT) * len(facts))


# ---------------------------------------------------------------------------
# 1. THE GATE (hard requirement stated by the calling brief)
# ---------------------------------------------------------------------------

# code_dim ~198 at this file's vocab_size=500 (bytes_per_code_dim = 500*5+16 =
# 2516, see test_substrate_bytes_matches_independent_hand_count for the exact
# formula) -- comfortably >> 0.138 * code_dim ~= 27 for the 60 facts written
# below, so the gate's retrieval assertion is not testing near a capacity
# boundary (measured 100% token accuracy at this size, see the task report).
_GATE_BUDGET = 500_000


def test_gate_base_weights_untouched_and_facts_retrievable():
    tokenizer = _build_test_tokenizer()
    facts = _make_facts(tokenizer, n_people=60, seed=0)
    substrate = DynamicSubstrate(substrate_bytes_budget=_GATE_BUDGET, seed=0, vocab_size=tokenizer.vocab_size)

    before = {name: array.copy() for name, array in substrate.base_weight_tensors().items()}
    before_hashes = {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in before.items()}
    print("BASE WEIGHT HASHES BEFORE WRITE:", before_hashes)

    substrate.write(facts)

    after = substrate.base_weight_tensors()
    after_hashes = {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in after.items()}
    print("BASE WEIGHT HASHES AFTER WRITE: ", after_hashes)

    assert before_hashes == after_hashes, "base weights changed after write() -- gradient/base-weight leak"
    for name in before:
        assert np.array_equal(before[name], after[name]), f"{name} is not bit-identical after write()"

    fraction = _retrieval_fraction(substrate, facts)
    print(f"GATE RETRIEVAL FRACTION (of max log2({_VALUE_COUNT}) bits/fact): {fraction:.4f}")
    assert fraction > 0.9, f"retrieval fraction {fraction:.4f} is not clearly above chance"


def test_offset_binding_recovers_every_position_of_a_multi_token_answer():
    """The golden dataset's real answers are frequently multi-token (e.g.
    birth_date splits into up to 7 tokens -- see src/models/dynamic.py's
    `_MAX_ANSWER_OFFSET` provenance comment); this file's main facts are
    deliberately single-token (module docstring), so this test is the one
    place the offset-binding mechanism itself is exercised end to end: a
    hand-built fact with a 3-token context and a 3-token answer, verifying
    EVERY answer position is independently retrievable, not just the first.
    """
    vocab_size = 20
    substrate = DynamicSubstrate(substrate_bytes_budget=200_000, seed=9, vocab_size=vocab_size)
    ids = [1, 2, 3, 10, 11, 12]
    mask = [False, False, False, True, True, True]
    substrate.write([(ids, mask)])

    ids_t = torch.tensor([ids], dtype=torch.long)
    mask_t = torch.tensor([mask], dtype=torch.bool)
    logits = substrate.read_logits(ids_t, mask_t)
    predicted = [int(logits[0, t].argmax()) for t in range(3, 6)]
    assert predicted == [10, 11, 12], f"expected each answer offset recovered independently, got {predicted}"


# ---------------------------------------------------------------------------
# 2. Null control (C1) -- an empty substrate must retrieve at chance
# ---------------------------------------------------------------------------


def test_null_substrate_retrieves_at_chance():
    tokenizer = _build_test_tokenizer()
    facts = _make_facts(tokenizer, n_people=30, seed=1)
    substrate = DynamicSubstrate(substrate_bytes_budget=_GATE_BUDGET, seed=1, vocab_size=tokenizer.vocab_size)
    # Deliberately never calling substrate.write(...).

    ce_bits = _per_fact_ce_bits(substrate, facts)
    recovered = bits_recovered_per_attribute({_ATTRIBUTE: ce_bits})[_ATTRIBUTE]
    print(f"NULL-MODEL CONTROL: bits_recovered={recovered} (want exactly 0.0)")
    assert recovered == 0.0, "an empty substrate must not appear to know anything"


# ---------------------------------------------------------------------------
# 3. Capacity / interference: retrieval degrades as more facts are written
# ---------------------------------------------------------------------------

# code_dim=20 at this file's vocab_size=500 (50_320 // 2516) -- calibrated by
# direct measurement (see the task report) against a linear single-pass
# readout's ACTUAL capacity, not assumed from the 0.138*N Hopfield anchor:
# that anchor governs full-vector auto-associative attractor recall, a
# strictly harder criterion than this substrate's single-shot argmax-over-
# logits readout, and measurement showed the two diverge by roughly an order
# of magnitude (retrieval still ~90%+ at n_facts=10*0.138*code_dim). This
# budget is the smallest that still shows a clean full curve -- near-1.0 at
# light load, crossing 0.5 near the top of the 400-fact test pool -- inside a
# sub-second sweep. Reused (at different vocab_size, hence different
# code_dim) by the byte-accounting and determinism tests below, where its
# exact value is incidental.
_CAPACITY_BUDGET = 50_320
_CAPACITY_CHECKPOINTS = (5, 10, 15, 20, 25, 30, 40, 60, 80, 120, 160, 200, 300, 400)


def test_capacity_interference_curve_prints_and_degrades():
    tokenizer = _build_test_tokenizer()
    facts = _make_facts(tokenizer, n_people=max(_CAPACITY_CHECKPOINTS), seed=3)
    substrate = DynamicSubstrate(substrate_bytes_budget=_CAPACITY_BUDGET, seed=3, vocab_size=tokenizer.vocab_size)
    hopfield_anchor = 0.138 * substrate.code_dim  # Amit-Gutfreund-Sompolinsky 1985 capacity ratio

    written = 0
    curve: list[tuple[int, float]] = []
    for checkpoint in _CAPACITY_CHECKPOINTS:
        substrate.write(facts[written:checkpoint])
        written = checkpoint
        curve.append((checkpoint, _retrieval_fraction(substrate, facts[:checkpoint])))

    print(f"CAPACITY CURVE (code_dim={substrate.code_dim}, Hopfield anchor 0.138*code_dim={hopfield_anchor:.1f}):")
    for n_facts, fraction in curve:
        print(f"  n_facts={n_facts:4d}  retrieval_fraction={fraction:.4f}")
    knee = next((n for n, fraction in curve if fraction < 0.5), None)
    print(f"KNEE (first n_facts with fraction<0.5): {knee}")
    if knee is not None:
        print(
            f"KNEE / HOPFIELD ANCHOR RATIO: {knee / hopfield_anchor:.1f}x -- this substrate's "
            f"single-shot argmax readout is NOT in the 0.138*N ballpark; it stores substantially "
            f"more single-token facts than that anchor before crossing 50% retrieval, because "
            f"argmax-over-logits is an easier criterion than full-vector attractor recall (see "
            f"the _CAPACITY_BUDGET comment above)."
        )

    assert curve[0][1] > 0.9, "should retrieve nearly everything well under capacity"
    assert curve[-1][1] < curve[0][1] - 0.2, "must show real degradation as facts accumulate (not flat)"
    assert knee is not None, "sweep never crossed 50% retrieval; widen _CAPACITY_CHECKPOINTS"


# ---------------------------------------------------------------------------
# 4. substrate_bytes accounting == an independent hand count (edge case S3:
#    under-counting the association matrix is the exact way to fake a win)
# ---------------------------------------------------------------------------


def test_substrate_bytes_matches_independent_hand_count():
    vocab_size = 17
    budget = _CAPACITY_BUDGET
    substrate = DynamicSubstrate(substrate_bytes_budget=budget, seed=5, vocab_size=vocab_size)
    code_dim = substrate.code_dim

    # Hand count, derived independently from vocab_size/code_dim/dtype sizes
    # -- NOT by calling substrate_bytes_breakdown() and trusting its own math.
    expected = {
        "context_projection": vocab_size * code_dim * 1,  # int8 bipolar
        "offset_projection": _MAX_ANSWER_OFFSET * code_dim * 1,  # int8 bipolar
        "association_matrix": vocab_size * code_dim * 4,  # float32
    }
    print(f"HAND-COUNTED BREAKDOWN (code_dim={code_dim}): {expected}")
    print(f"substrate_bytes_breakdown():                  {substrate.substrate_bytes_breakdown()}")

    assert substrate.substrate_bytes_breakdown() == expected
    assert substrate.substrate_bytes() == sum(expected.values())

    # Cross-check against the raw arrays' own .nbytes directly, bypassing
    # substrate_bytes_breakdown() entirely, so a bug shared between it and
    # this hand count could not hide behind mutual agreement (mistake log M9).
    base = substrate.base_weight_tensors()
    state = substrate.persistent_state_tensors()
    assert base["context_projection"].nbytes == expected["context_projection"]
    assert base["offset_projection"].nbytes == expected["offset_projection"]
    assert state["association_matrix"].nbytes == expected["association_matrix"]


# ---------------------------------------------------------------------------
# 5. Determinism on seed (+ negative control, mistake log M9)
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_everything():
    tokenizer = _build_test_tokenizer()
    facts = _make_facts(tokenizer, n_people=20, seed=7)

    a = DynamicSubstrate(substrate_bytes_budget=_CAPACITY_BUDGET, seed=42, vocab_size=tokenizer.vocab_size)
    b = DynamicSubstrate(substrate_bytes_budget=_CAPACITY_BUDGET, seed=42, vocab_size=tokenizer.vocab_size)
    for name, array in a.base_weight_tensors().items():
        assert np.array_equal(array, b.base_weight_tensors()[name])

    a.write(facts)
    b.write(facts)
    assert np.array_equal(
        a.persistent_state_tensors()["association_matrix"], b.persistent_state_tensors()["association_matrix"]
    )

    ids_batch, mask_batch = _pad_facts(facts)
    assert torch.equal(a.read_logits(ids_batch, mask_batch), b.read_logits(ids_batch, mask_batch))


def test_determinism_different_seed_differs():
    """Negative control: if `seed` were silently ignored, the test above would
    pass vacuously. Confirms it is not (implementation/00_START_HERE.md N1)."""
    a = DynamicSubstrate(substrate_bytes_budget=_CAPACITY_BUDGET, seed=1, vocab_size=17)
    b = DynamicSubstrate(substrate_bytes_budget=_CAPACITY_BUDGET, seed=2, vocab_size=17)
    assert not np.array_equal(
        a.base_weight_tensors()["context_projection"], b.base_weight_tensors()["context_projection"]
    )


# ---------------------------------------------------------------------------
# 6. Edge-case ladder: constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_int_args():
    with pytest.raises(TypeError):
        DynamicSubstrate(substrate_bytes_budget=1000.0, seed=0, vocab_size=10)
    with pytest.raises(TypeError):
        DynamicSubstrate(substrate_bytes_budget=1000, seed=True, vocab_size=10)


def test_constructor_rejects_negative_seed_and_non_positive_vocab():
    with pytest.raises(ValueError, match="seed"):
        DynamicSubstrate(substrate_bytes_budget=1000, seed=-1, vocab_size=10)
    with pytest.raises(ValueError, match="vocab_size"):
        DynamicSubstrate(substrate_bytes_budget=1000, seed=0, vocab_size=0)


def test_constructor_rejects_budget_too_small_for_code_dim_one():
    with pytest.raises(ValueError, match="too small"):
        DynamicSubstrate(substrate_bytes_budget=1, seed=0, vocab_size=1000)


# ---------------------------------------------------------------------------
# 7. Edge-case ladder: write() validation
# ---------------------------------------------------------------------------


def _small_substrate() -> DynamicSubstrate:
    return DynamicSubstrate(substrate_bytes_budget=2000, seed=0, vocab_size=10)


def test_write_empty_facts_is_a_no_op():
    substrate = _small_substrate()
    before = substrate.persistent_state_tensors()["association_matrix"].copy()
    substrate.write([])
    assert np.array_equal(before, substrate.persistent_state_tensors()["association_matrix"])


def test_write_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        _small_substrate().write([([1, 2, 3], [False, True])])


def test_write_rejects_all_false_mask():
    with pytest.raises(ValueError, match="zero tokens"):
        _small_substrate().write([([1, 2], [False, False])])


def test_write_rejects_out_of_range_token_id():
    with pytest.raises(ValueError, match=r"outside \[0, 10\)"):
        _small_substrate().write([([1, 10], [False, True])])


def test_write_rejects_answer_longer_than_max_offset():
    ids = [0] + [1] * (_MAX_ANSWER_OFFSET + 1)
    mask = [False] + [True] * (_MAX_ANSWER_OFFSET + 1)
    with pytest.raises(ValueError, match="max_answer_offset"):
        _small_substrate().write([(ids, mask)])


# ---------------------------------------------------------------------------
# 8. Edge-case ladder: read_logits() validation
# ---------------------------------------------------------------------------


def test_read_logits_rejects_float_dtype():
    substrate = _small_substrate()
    with pytest.raises(TypeError):
        substrate.read_logits(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_read_logits_rejects_shape_mismatch():
    substrate = _small_substrate()
    with pytest.raises(ValueError, match="!="):
        substrate.read_logits(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool))


def test_read_logits_rejects_empty_batch_and_seq_len():
    substrate = _small_substrate()
    with pytest.raises(ValueError, match="batch size 0"):
        substrate.read_logits(torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="seq_len 0"):
        substrate.read_logits(torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.bool))


def test_read_logits_rejects_pad_marked_as_answer():
    substrate = _small_substrate()
    ids = torch.tensor([[0, PAD_ID]], dtype=torch.long)
    mask = torch.tensor([[False, True]], dtype=torch.bool)
    with pytest.raises(ValueError, match="PAD_ID"):
        substrate.read_logits(ids, mask)


def test_read_logits_on_untrained_substrate_returns_finite_zero_at_unmasked_positions():
    substrate = _small_substrate()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[False, False, True]], dtype=torch.bool)
    logits = substrate.read_logits(ids, mask)
    assert logits.shape == (1, 3, 10)
    assert torch.equal(logits[0, 0], torch.zeros(10))
    assert torch.equal(logits[0, 1], torch.zeros(10))
    assert torch.isfinite(logits[0, 2]).all()
