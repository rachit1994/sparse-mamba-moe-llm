"""Tests for the knowledge-bits-recovered metric (src/metrics/bits.py).

Read implementation/02_testing_philosophy.md before extending this file: a test
that only passes when the code works is worth very little. Every test here is
designed to FAIL if the specific bug it targets is present -- see the
"deliberate breakage" verification log in the task report for evidence that
each one actually does.

Where possible, results are cross-checked against the lead's independent
numpy-only reference oracle at tests/lead/reference_bits.py (LEAD-OWNED, never
modified here): two independent implementations agreeing on non-trivial inputs
is real evidence; a single implementation with a green suite is not
(implementation/02_testing_philosophy.md section 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.data.schema import ATTRIBUTE_CARDINALITY, ATTRIBUTES, bits_per_attribute
from src.metrics.bits import (
    EntropyBoundViolation,
    answer_cross_entropy_bits,
    bits_per_parameter,
    bits_recovered_per_attribute,
    count_parameters,
)
from tests.lead.reference_bits import answer_ce_bits as reference_answer_ce_bits
from tests.lead.reference_bits import bits_recovered as reference_bits_recovered

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_logits(vocab_size: int, n_tokens: int = 1) -> torch.Tensor:
    """Logits with p(any class) == 1/vocab_size exactly: the chance baseline."""
    return torch.zeros(n_tokens, vocab_size)


def _confident_logits(vocab_size: int, correct_idx: int, n_tokens: int = 1, gap: float = 60.0) -> torch.Tensor:
    """Logits that put ~all probability mass on `correct_idx`."""
    logits = torch.zeros(n_tokens, vocab_size)
    logits[:, correct_idx] = gap
    return logits


# ---------------------------------------------------------------------------
# 1. Chance-level: p(correct) == 1/V -> bits_recovered == 0
# ---------------------------------------------------------------------------


def test_chance_level_answer_cross_entropy_equals_log2_v():
    """A uniform model's CE_bits on one answer token is exactly log2(V)."""
    vocab_size = 100  # matches the 'major' attribute's cardinality
    logits = _uniform_logits(vocab_size)
    targets = torch.tensor([7])
    mask = torch.tensor([True])

    ce_bits = answer_cross_entropy_bits(logits, targets, mask)

    assert ce_bits == pytest.approx(math.log2(vocab_size), abs=1e-4)


def test_chance_level_bits_recovered_is_zero():
    """A model at exactly chance recovers 0 bits for an attribute, across many people."""
    attribute = "major"
    vocab_size = ATTRIBUTE_CARDINALITY[attribute]
    n_people = 25

    ce_values = [
        answer_cross_entropy_bits(
            _uniform_logits(vocab_size), torch.tensor([i % vocab_size]), torch.tensor([True])
        )
        for i in range(n_people)
    ]

    recovered = bits_recovered_per_attribute({attribute: ce_values})

    print(f"[chance-level] bits_recovered({attribute}) over {n_people} people = {recovered[attribute]!r}")
    assert recovered[attribute] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Perfect model: p(correct) ~= 1.0 -> bits_recovered ~= log2(V) per item
# ---------------------------------------------------------------------------


def test_perfect_model_answer_cross_entropy_near_zero():
    vocab_size = 1_000  # matches 'birth_city' / 'employer'
    logits = _confident_logits(vocab_size, correct_idx=42)
    targets = torch.tensor([42])
    mask = torch.tensor([True])

    ce_bits = answer_cross_entropy_bits(logits, targets, mask)

    assert ce_bits == pytest.approx(0.0, abs=1e-6)


def test_perfect_model_bits_recovered_near_log2_v_per_person():
    attribute = "employer"
    vocab_size = ATTRIBUTE_CARDINALITY[attribute]
    n_people = 25
    ceiling = math.log2(vocab_size)

    ce_values = [
        answer_cross_entropy_bits(
            _confident_logits(vocab_size, correct_idx=i % vocab_size),
            torch.tensor([i % vocab_size]),
            torch.tensor([True]),
        )
        for i in range(n_people)
    ]

    recovered = bits_recovered_per_attribute({attribute: ce_values})

    print(
        f"[perfect-model] bits_recovered({attribute}) over {n_people} people = "
        f"{recovered[attribute]!r}, expected ~= {n_people * ceiling!r}"
    )
    assert recovered[attribute] == pytest.approx(n_people * ceiling, abs=1e-3)
    # per-person average must land on log2(V), not some scaled/averaged variant
    assert recovered[attribute] / n_people == pytest.approx(ceiling, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. NATS-vs-BITS regression -- the most important test in this file.
#    Must FAIL if the /math.log(2) conversion is removed. Verified by
#    deliberate breakage in the task report; not re-derived from the
#    implementation under test (uses an independent hand-computation and the
#    lead's independent numpy oracle).
# ---------------------------------------------------------------------------


def test_nats_vs_bits_regression_hand_computed():
    """V=2, single answer token: p(correct) = 3/4 by construction.

    nats = -ln(3/4) = ln(4/3) ~= 0.287682
    bits = ln(4/3) / ln(2)   ~= 0.415037

    Computed here by hand from softmax first principles -- NOT by calling
    F.cross_entropy independently -- so this test cannot pass merely because
    the implementation is internally self-consistent.
    """
    logits = torch.tensor([[0.0, math.log(3.0)]])
    targets = torch.tensor([1])
    mask = torch.tensor([True])

    expected_nats = math.log(4.0 / 3.0)
    expected_bits = expected_nats / math.log(2.0)

    actual_bits = answer_cross_entropy_bits(logits, targets, mask)

    print(f"[nats-vs-bits] expected_nats={expected_nats!r} expected_bits={expected_bits!r} actual={actual_bits!r}")

    # Tight enough that a missing /ln(2) (a 1.4427x error) cannot slip through:
    # expected_nats and expected_bits differ by 44%, this tolerance is 0.01%.
    assert actual_bits == pytest.approx(expected_bits, rel=1e-4)
    # Explicitly rule out the un-converted (nats) value being mistaken for correct.
    assert actual_bits != pytest.approx(expected_nats, rel=1e-2)


def test_nats_vs_bits_regression_agrees_with_independent_numpy_oracle():
    """Cross-check against tests/lead/reference_bits.py (independent, no torch)."""
    logits = torch.tensor([[0.0, math.log(3.0)]])
    targets = torch.tensor([1])
    mask = torch.tensor([True])

    actual_bits = answer_cross_entropy_bits(logits, targets, mask)
    oracle_bits = reference_answer_ce_bits(
        logits.numpy().astype(np.float64), targets.numpy(), mask.numpy()
    )

    print(f"[oracle-agreement] mine={actual_bits!r} oracle={oracle_bits!r}")
    assert actual_bits == pytest.approx(oracle_bits, rel=1e-5)


def test_reduction_is_sum_not_mean_over_multiple_answer_tokens():
    """Two answer tokens: one at chance (2 bits), one near-perfect (~0 bits).

    Summed CE_bits must be ~= 2.0 (2.0 + 0.0). If sum were silently replaced by
    mean, this would read ~= 1.0 instead -- clearly distinguishable.
    """
    vocab_size = 4
    logits = torch.stack(
        [
            _uniform_logits(vocab_size)[0],  # token 0: chance, CE_bits = 2.0
            _confident_logits(vocab_size, correct_idx=1)[0],  # token 1: near-perfect, CE_bits ~= 0.0
        ]
    )
    targets = torch.tensor([2, 1])
    mask = torch.tensor([True, True])

    total_bits = answer_cross_entropy_bits(logits, targets, mask)

    print(f"[sum-not-mean] total_bits={total_bits!r}, expected ~= 2.0 (sum), NOT ~= 1.0 (mean)")
    assert total_bits == pytest.approx(2.0, abs=1e-4)
    assert total_bits != pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# 4. Masking: prompt tokens must not affect the score; answer tokens must.
# ---------------------------------------------------------------------------


def test_masking_prompt_tokens_do_not_affect_score():
    vocab_size = 5
    seq_len = 6
    answer_mask = torch.tensor([False, False, False, True, True, True])

    torch.manual_seed(1)
    logits_a = torch.randn(seq_len, vocab_size)
    targets_a = torch.tensor([0, 1, 2, 3, 4, 0])

    # Different prompt-region (first 3) logits/targets, identical answer region.
    logits_b = logits_a.clone()
    logits_b[:3] = torch.randn(3, vocab_size) * 100  # wildly different prompt logits
    targets_b = targets_a.clone()
    targets_b[:3] = torch.tensor([4, 3, 2])  # different prompt targets

    score_a = answer_cross_entropy_bits(logits_a, targets_a, answer_mask)
    score_b = answer_cross_entropy_bits(logits_b, targets_b, answer_mask)

    print(f"[masking] score with prompt A = {score_a!r}, score with prompt B = {score_b!r}")
    assert score_a == score_b


def test_masking_answer_tokens_do_change_score():
    vocab_size = 5
    seq_len = 6
    answer_mask = torch.tensor([False, False, False, True, True, True])

    torch.manual_seed(2)
    logits = torch.randn(seq_len, vocab_size)
    targets_a = torch.tensor([0, 1, 2, 3, 4, 0])
    targets_b = targets_a.clone()
    targets_b[4] = (targets_b[4] + 1) % vocab_size  # mutate one answer-region target

    score_a = answer_cross_entropy_bits(logits, targets_a, answer_mask)
    score_b = answer_cross_entropy_bits(logits, targets_b, answer_mask)

    print(f"[masking] score before mutation = {score_a!r}, after answer mutation = {score_b!r}")
    assert score_a != score_b


def test_masking_ignores_nonfinite_garbage_in_prompt_region():
    """Non-finite values are allowed outside the answer mask -- they are never touched."""
    vocab_size = 5
    logits = torch.zeros(4, vocab_size)
    logits[0, 0] = float("nan")
    logits[1, 0] = float("inf")
    logits[2:] = _confident_logits(vocab_size, correct_idx=2, n_tokens=2)
    targets = torch.tensor([0, 0, 2, 2])
    mask = torch.tensor([False, False, True, True])

    # Must not raise, and must equal the score computed on the clean subsequence alone.
    score = answer_cross_entropy_bits(logits, targets, mask)
    clean_score = answer_cross_entropy_bits(logits[2:], targets[2:], mask[2:])
    assert score == pytest.approx(clean_score, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Entropy bound: the <= entropy assertion must actually raise when violated.
# ---------------------------------------------------------------------------


def test_entropy_bound_raises_on_negative_ce_bits():
    """A negative CE_bits is impossible for real cross-entropy; must raise."""
    with pytest.raises(EntropyBoundViolation):
        bits_recovered_per_attribute({"major": [-1.0]})


def test_entropy_bound_raises_on_nonfinite_ce_bits():
    with pytest.raises(EntropyBoundViolation):
        bits_recovered_per_attribute({"birth_city": [float("nan")]})
    with pytest.raises(EntropyBoundViolation):
        bits_recovered_per_attribute({"birth_city": [float("inf")]})


def test_entropy_bound_error_names_the_offending_attribute_and_index():
    with pytest.raises(EntropyBoundViolation, match="university"):
        bits_recovered_per_attribute({"university": [1.0, 2.0, -0.5]})


def test_entropy_bound_does_not_raise_for_legitimate_worse_than_chance():
    """A model worse than chance (CE_bits > log2(V), but still >= 0) is legitimate: clamp, don't raise."""
    vocab_size = ATTRIBUTE_CARDINALITY["major"]
    worse_than_chance_ce = math.log2(vocab_size) + 5.0  # legitimate, if bad, cross-entropy
    recovered = bits_recovered_per_attribute({"major": [worse_than_chance_ce]})
    assert recovered["major"] == pytest.approx(0.0, abs=1e-9)


def test_entropy_bound_agrees_with_independent_numpy_oracle():
    """The reference implementation's own ceiling check must also fire for the same input."""
    with pytest.raises(ValueError):
        reference_bits_recovered(cardinality=ATTRIBUTE_CARDINALITY["major"], ce_bits_per_item=[-1.0])


# ---------------------------------------------------------------------------
# 6. Determinism: identical inputs -> identical float output.
# ---------------------------------------------------------------------------


def test_determinism_answer_cross_entropy_bits():
    torch.manual_seed(42)
    logits = torch.randn(16, 8, 200)
    targets = torch.randint(0, 200, (16, 8))
    mask = torch.zeros(16, 8, dtype=torch.bool)
    mask[:, 4:] = True

    run_1 = answer_cross_entropy_bits(logits, targets, mask)
    run_2 = answer_cross_entropy_bits(logits.clone(), targets.clone(), mask.clone())

    assert run_1 == run_2  # bit-identical, not approximately equal


def test_determinism_bits_recovered_per_attribute():
    ce_values = [1.0, 2.5, 0.3, 4.4, 6.0]
    r1 = bits_recovered_per_attribute({"employer": list(ce_values)})
    r2 = bits_recovered_per_attribute({"employer": list(ce_values)})
    assert r1 == r2


# ---------------------------------------------------------------------------
# Contract / shape / type validation
# ---------------------------------------------------------------------------


def test_empty_answer_mask_raises():
    logits = torch.zeros(3, 5)
    targets = torch.tensor([0, 1, 2])
    mask = torch.tensor([False, False, False])
    with pytest.raises(ValueError, match="zero tokens"):
        answer_cross_entropy_bits(logits, targets, mask)


def test_shape_mismatch_reports_both_shapes():
    logits = torch.zeros(3, 5)
    targets = torch.tensor([0, 1])  # wrong length
    mask = torch.tensor([True, True])
    with pytest.raises(ValueError) as excinfo:
        answer_cross_entropy_bits(logits, targets, mask)
    message = str(excinfo.value)
    assert "(3, 5)" in message
    assert "(2,)" in message


def test_targets_out_of_range_raises():
    logits = torch.zeros(1, 5)
    targets = torch.tensor([5])  # vocab_size == 5, valid range is [0, 5)
    mask = torch.tensor([True])
    with pytest.raises(ValueError, match="out of range"):
        answer_cross_entropy_bits(logits, targets, mask)


def test_nonfinite_logits_on_answer_token_raises():
    logits = torch.zeros(1, 5)
    logits[0, 0] = float("inf")
    targets = torch.tensor([0])
    mask = torch.tensor([True])
    with pytest.raises(ValueError, match="non-finite"):
        answer_cross_entropy_bits(logits, targets, mask)


def test_float_targets_raise_typeerror():
    logits = torch.zeros(1, 5)
    targets = torch.tensor([1.0])  # should be integer token ids
    mask = torch.tensor([True])
    with pytest.raises(TypeError):
        answer_cross_entropy_bits(logits, targets, mask)


def test_unknown_attribute_raises():
    with pytest.raises(ValueError, match="unknown attribute"):
        bits_recovered_per_attribute({"not_a_real_attribute": [1.0]})


def test_empty_attribute_list_raises():
    with pytest.raises(ValueError, match="empty"):
        bits_recovered_per_attribute({"major": []})


# ---------------------------------------------------------------------------
# bits_per_parameter
# ---------------------------------------------------------------------------


def test_bits_per_parameter_basic():
    assert bits_per_parameter(200.0, 100) == pytest.approx(2.0)


@pytest.mark.parametrize("bad_count", [0, -1, -1_000_000])
def test_bits_per_parameter_nonpositive_count_raises(bad_count):
    with pytest.raises(ValueError):
        bits_per_parameter(10.0, bad_count)


def test_bits_per_parameter_negative_total_raises():
    with pytest.raises(ValueError):
        bits_per_parameter(-1.0, 100)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


class _TinyModelWithTiedEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int = 10, hidden: int = 4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.linear = torch.nn.Linear(hidden, hidden, bias=True)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying


def test_count_parameters_include_embeddings_true_by_default():
    model = _TinyModelWithTiedEmbedding(vocab_size=10, hidden=4)
    total = count_parameters(model)
    # embed(10*4) + linear.weight(4*4) + linear.bias(4)  [lm_head.weight tied to embed, not double-counted]
    expected = (10 * 4) + (4 * 4) + 4
    assert total == expected


def test_count_parameters_excludes_embeddings_when_requested():
    model = _TinyModelWithTiedEmbedding(vocab_size=10, hidden=4)
    total_with = count_parameters(model, include_embeddings=True)
    total_without = count_parameters(model, include_embeddings=False)
    assert total_without == total_with - (10 * 4)


def test_count_parameters_excludes_frozen_parameters():
    model = _TinyModelWithTiedEmbedding(vocab_size=10, hidden=4)
    model.linear.bias.requires_grad = False
    total = count_parameters(model)
    expected = (10 * 4) + (4 * 4)  # linear.bias (4) excluded
    assert total == expected


def test_count_parameters_empty_model_is_zero():
    assert count_parameters(torch.nn.Module()) == 0


def test_count_parameters_rejects_non_module():
    with pytest.raises(TypeError):
        count_parameters("not a model")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Property: bits_recovered_per_attribute never exceeds the schema-derived
# entropy ceiling for legitimate (non-negative, finite) CE_bits inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_property_bits_recovered_never_exceeds_schema_entropy(attribute):
    rng = np.random.default_rng(0)
    vocab_size = ATTRIBUTE_CARDINALITY[attribute]
    n_people = 50
    # CE_bits sampled from [0, 3*log2(V)] to cover chance, perfect, and worse-than-chance.
    ce_values = list(rng.uniform(0.0, 3.0 * math.log2(vocab_size), size=n_people))

    recovered = bits_recovered_per_attribute({attribute: ce_values})

    ceiling = n_people * bits_per_attribute()[attribute]
    assert 0.0 <= recovered[attribute] <= ceiling + 1e-6


# ---------------------------------------------------------------------------
# Cross-check: independent numpy oracle agreement on random inputs.
# ---------------------------------------------------------------------------


def test_random_inputs_agree_with_independent_numpy_oracle():
    rng = np.random.default_rng(7)
    vocab_size = 37
    seq_len = 9

    for _ in range(10):
        logits_np = rng.normal(size=(seq_len, vocab_size)).astype(np.float64)
        targets_np = rng.integers(0, vocab_size, size=seq_len)
        mask_np = rng.random(seq_len) < 0.5
        if not mask_np.any():
            mask_np[0] = True

        mine = answer_cross_entropy_bits(
            torch.from_numpy(logits_np).float(),
            torch.from_numpy(targets_np).long(),
            torch.from_numpy(mask_np),
        )
        oracle = reference_answer_ce_bits(logits_np, targets_np, mask_np)

        assert mine == pytest.approx(oracle, rel=1e-3, abs=1e-3)
