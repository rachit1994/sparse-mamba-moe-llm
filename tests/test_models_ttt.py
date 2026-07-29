"""Tests for the TTT-Linear arm (src/models/ttt.py).

THE QUESTION this file answers, and HOW TO REVIEW IT (implementation/10_code_style.md):
every test below is designed to FAIL if the specific property it names does
not hold -- see the task report's break/revert log for real captured output
proving each one actually does. A test that cannot fail when its target is
broken is itself a defect (implementation/09_mistake_log.md M6/M7); none of
these are trusted until shown to fail against the break they exist to catch.

Test-to-requirement map (matches the calling brief's 7 required tests):
  1. Gradient correctness:  test_gradient_matches_autograd_within_tolerance
  2. Inner loop learns:     test_inner_loop_loss_decreases_monotonically_on_repeated_token
  3. Base weights untouched: test_base_weights_untouched_after_forward_sequence
  4. State is load-bearing: test_state_is_load_bearing_reset_changes_output
  5. Loss improves with context length: test_loss_improves_with_context_length_on_repeating_structure
  6. substrate_bytes accounting: test_substrate_bytes_matches_independent_hand_count
  7. Determinism: test_determinism_same_seed_identical_everything (+ negative
     control test_determinism_different_seed_differs -- implementation/09_mistake_log.md
     M9: agreement on a value is only evidence if it is hard to produce by
     accident, so a determinism test needs a "seeds differ" counterpart).

File structure follows implementation/10_code_style.md, adapted the same way
its own reference sibling test file (tests/test_models_dynamic.py) adapts it
for a pytest file rather than a `verify_*.py` report script: constants with
provenance, then pure helper functions (one quantity each, no printing), then
the tests themselves, which both assert (self-check) and print (report) --
pytest's `-s` flag surfaces the printed evidence this task's report quotes.

CONTAMINATION / hyperparameter honesty: `_DIM` and `_INNER_LR` are each
defined ONCE below and used by every test in this file, so no single test's
pass/fail was tuned in isolation to make that test pass (implementation/00_START_HERE.md
N3: loosening a test to get green destroys the experiment). `_INNER_LR`'s
provenance comment states the convexity argument for why it works, not just
that it was observed to.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from src.models.ttt import TTTLinear, _inner_grad_linear, _inner_loss_linear  # noqa: SLF001 -- gradient-correctness test 1 exercises these directly, by design

# ---------------------------------------------------------------------------
# SECTION 1 -- CONSTANTS (every value below is used by name, nowhere inlined)
# ---------------------------------------------------------------------------

# Shared across every test that constructs a TTTLinear, so no individual
# test's hyperparameters were hand-picked to make that one test pass.
# dim=8 is arbitrary-but-small: it keeps test 5's 64+256+1024+4096=5,440-step
# sweep fast on 4 CPU cores. inner_lr=0.02: __init__ fan-in-normalises
# theta_K/Q/V by 1/sqrt(dim), which keeps k = theta_K @ x at O(1) per-entry
# variance regardless of dim; TTT-Linear's inner loss is a strictly convex
# quadratic in W for fixed (k, v) (see test 2's docstring), so constant-step
# gradient descent provably does not increase the loss for any
# inner_lr < 1/L, L = 2*||k||^2 -- 0.02 is comfortably inside that region for
# the O(1)-scaled k this init produces, which is WHY tests 2 and 5 decrease,
# not merely an observation that they happen to.
_DIM = 8
_INNER_LR = 0.02
_BASE_SEED = 1

# float32 (this project's stated floor-and-ceiling precision, src/metrics/bits.py)
_BYTES_PER_FLOAT32 = 4

# Test 1 -- gradient correctness. Dims span well below and above _DIM so the
# analytic-vs-autograd check is not accidentally validated only at one size.
_GRADIENT_CHECK_DIMS = (1, 2, 4, 8, 16, 32, 64)
_GRADIENT_CHECK_SEED = 42
_GRADIENT_TOLERANCE = 1e-5  # task specification's stated bound

# Test 2 -- repeated-token inner loop. 30 steps is enough for float32 to
# reach its own noise floor on this convex problem (measured: loss hits
# exactly 0.0 well before step 30 at _DIM/_INNER_LR above), which is itself
# evidence the loop is actually minimising something, not simply oscillating.
_REPEATED_TOKEN_STEPS = 30
_REPEATED_TOKEN_X_SEED = 777

# Test 3 -- base weights untouched. 500 is "many tokens" per the brief: two
# orders of magnitude past what a single-fact test would need, so a leak that
# only appears after sustained use is not missed.
_MANY_TOKENS_STEPS = 500
_MANY_TOKENS_SEED = 9

# Test 4 -- state is load-bearing.
_WARMUP_STEPS = 50
_WARMUP_SEED = 3
_PROBE_STEPS = 5
_PROBE_SEED = 4

# Test 5 -- the paper's headline property. Context lengths are the brief's
# exact list. _CONTEXT_LAST_K=32 is the window "mean inner loss over the last
# k tokens" is averaged over; period=8 makes the sequence "repeating
# structure" (a fixed 8-token pattern retiled to fill each length) without
# being so short that one period is trivially memorised in a single step.
_CONTEXT_LENGTHS = (64, 256, 1024, 4096)
_CONTEXT_LAST_K = 32
_CONTEXT_PERIOD = 8
_CONTEXT_PATTERN_SEED = 12345

# Test 7 -- determinism.
_DETERMINISM_SEED_A = 555
_DETERMINISM_SEED_B = 556  # different from A: the negative control (M9)
_DETERMINISM_STEPS = 40
_DETERMINISM_X_SEED = 6


# ---------------------------------------------------------------------------
# SECTION 2 -- PURE FUNCTIONS (one quantity each; no printing, no assertions)
# ---------------------------------------------------------------------------


def _repeating_sequence(period: int, length: int, dim: int, seed: int) -> torch.Tensor:
    """A length-`length` sequence built by retiling a fixed `period`-token random pattern."""
    generator = torch.Generator().manual_seed(seed)
    pattern = torch.randn(period, dim, generator=generator, dtype=torch.float32)
    repeats = -(-length // period)  # ceiling division, no float/round
    return pattern.repeat(repeats, 1)[:length]


def _mean_loss_last_k(trace: torch.Tensor, k: int) -> float:
    """Mean of the last `k` entries of a 1-D loss trace (or all of it if shorter than `k`)."""
    window = min(k, trace.numel())
    return trace[-window:].mean().item()


def _hash_tensor(tensor: torch.Tensor) -> str:
    """A stable content hash of a float32 tensor's exact bits."""
    return hashlib.sha256(tensor.contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def _hash_base_state(ttt: TTTLinear) -> dict[str, str]:
    """One hash per base-weight tensor, keyed the same as `TTTLinear.base_state_dict`."""
    return {name: _hash_tensor(tensor) for name, tensor in ttt.base_state_dict().items()}


def _gradient_check_trials(dims: tuple[int, ...], seed: int) -> list[tuple[int, float]]:
    """For each `dim`, one random (W, k, v) trial: `(dim, max|analytic - autograd|)`.

    Independent verification of `_inner_grad_linear`'s formula: `W` is given
    `requires_grad=True` and `_inner_loss_linear(W, k, v).backward()` is used
    to obtain `torch.autograd`'s gradient, which is compared against
    `_inner_grad_linear`'s closed form on the SAME inputs.
    """
    generator = torch.Generator().manual_seed(seed)
    results: list[tuple[int, float]] = []
    for dim in dims:
        W = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
        k = torch.randn(dim, generator=generator, dtype=torch.float32)
        v = torch.randn(dim, generator=generator, dtype=torch.float32)

        W_autograd = W.clone().requires_grad_(True)
        _inner_loss_linear(W_autograd, k, v).backward()
        autograd_grad = W_autograd.grad
        assert autograd_grad is not None  # backward() on a scalar always populates .grad

        analytic_grad = _inner_grad_linear(W, k, v)
        max_abs_diff = (analytic_grad - autograd_grad).abs().max().item()
        results.append((dim, max_abs_diff))
    return results


# ---------------------------------------------------------------------------
# 1. GRADIENT CORRECTNESS -- the test that proves this is TTT, not something else
# ---------------------------------------------------------------------------


def test_gradient_matches_autograd_within_tolerance():
    results = _gradient_check_trials(_GRADIENT_CHECK_DIMS, seed=_GRADIENT_CHECK_SEED)
    print("ANALYTIC 2(Wk-v)k^T vs torch.autograd, max|diff| per dim:")
    for dim, max_abs_diff in results:
        print(f"  dim={dim:3d}  max_abs_diff={max_abs_diff:.3e}")
    for dim, max_abs_diff in results:
        assert max_abs_diff < _GRADIENT_TOLERANCE, (
            f"dim={dim}: analytic inner gradient 2(Wk-v)k^T diverges from "
            f"torch.autograd by {max_abs_diff} >= tolerance {_GRADIENT_TOLERANCE}"
        )


# ---------------------------------------------------------------------------
# 2. THE INNER LOOP LEARNS
# ---------------------------------------------------------------------------


def test_inner_loop_loss_decreases_monotonically_on_repeated_token():
    """Same token every step makes l(W) a fixed strictly-convex quadratic in W.

    With `k`, `v` constant across steps, `l(W) = ||Wk - v||^2` has Hessian
    `2 k k^T` (w.r.t. each row of `W`), positive semi-definite with a single
    nonzero eigenvalue `L = 2||k||^2`. Constant-step gradient descent on a
    convex quadratic does not increase the objective for any step size in
    `(0, 2/L)` -- `_INNER_LR`'s provenance comment (Section 1) is why that
    holds here without per-test tuning.
    """
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_BASE_SEED)
    x = torch.randn(1, _DIM, generator=torch.Generator().manual_seed(_REPEATED_TOKEN_X_SEED))
    xs = x.repeat(_REPEATED_TOKEN_STEPS, 1)

    ttt.forward_sequence(xs)
    trace = ttt.inner_loss_trace.tolist()
    print(f"INNER LOSS SEQUENCE over {_REPEATED_TOKEN_STEPS} repeats of one token:")
    print("  " + ", ".join(f"{v:.6g}" for v in trace))

    diffs = [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]
    assert all(d <= 1e-9 for d in diffs), f"inner loss failed to decrease monotonically: {trace}"
    assert trace[0] > trace[-1] > -1e-9, f"no net decrease over the run: {trace}"


# ---------------------------------------------------------------------------
# 3. BASE WEIGHTS UNTOUCHED
# ---------------------------------------------------------------------------


def test_base_weights_untouched_after_forward_sequence():
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_BASE_SEED)
    before_hashes = _hash_base_state(ttt)
    print("BASE WEIGHT HASHES BEFORE forward_sequence:", before_hashes)

    xs = torch.randn(_MANY_TOKENS_STEPS, _DIM, generator=torch.Generator().manual_seed(_MANY_TOKENS_SEED))
    ttt.forward_sequence(xs)

    after_hashes = _hash_base_state(ttt)
    print("BASE WEIGHT HASHES AFTER  forward_sequence:", after_hashes)
    assert before_hashes == after_hashes, (
        f"theta_K/Q/V changed after {_MANY_TOKENS_STEPS} tokens of forward_sequence "
        f"-- the inner loop is leaking a gradient into the base weights"
    )


# ---------------------------------------------------------------------------
# 4. STATE IS LOAD-BEARING
# ---------------------------------------------------------------------------


def test_state_is_load_bearing_reset_changes_output():
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_BASE_SEED)
    warmup = torch.randn(_WARMUP_STEPS, _DIM, generator=torch.Generator().manual_seed(_WARMUP_SEED))
    probe = torch.randn(_PROBE_STEPS, _DIM, generator=torch.Generator().manual_seed(_PROBE_SEED))

    ttt.forward_sequence(warmup)
    warm_output = ttt.forward_sequence(probe.clone())

    ttt.reset_state()
    cold_output = ttt.forward_sequence(probe.clone())

    print("WARM-STATE output (after 50-token warmup):", warm_output.flatten()[:4].tolist(), "...")
    print("COLD-STATE output (after reset_state):    ", cold_output.flatten()[:4].tolist(), "...")
    assert not torch.allclose(warm_output, cold_output), (
        "reset_state() had no effect on output for the same probe sequence -- "
        "the inner state carries no information, so this is not TTT"
    )


# ---------------------------------------------------------------------------
# 5. THE PAPER'S HEADLINE PROPERTY -- loss improves as context grows
# ---------------------------------------------------------------------------


def test_loss_improves_with_context_length_on_repeating_structure():
    means: list[float] = []
    for length in _CONTEXT_LENGTHS:
        xs = _repeating_sequence(period=_CONTEXT_PERIOD, length=length, dim=_DIM, seed=_CONTEXT_PATTERN_SEED)
        ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_BASE_SEED)  # fresh state per length
        ttt.forward_sequence(xs)
        means.append(_mean_loss_last_k(ttt.inner_loss_trace, _CONTEXT_LAST_K))

    print(f"MEAN INNER LOSS OVER LAST {_CONTEXT_LAST_K} TOKENS vs CONTEXT LENGTH:")
    for length, mean_loss in zip(_CONTEXT_LENGTHS, means):
        print(f"  length={length:5d}  mean_loss={mean_loss:.6f}")

    for (earlier_length, earlier_mean), (later_length, later_mean) in zip(
        zip(_CONTEXT_LENGTHS, means), zip(_CONTEXT_LENGTHS[1:], means[1:])
    ):
        assert later_mean <= earlier_mean + 1e-9, (
            f"loss did not keep improving with context length: "
            f"length={earlier_length} mean={earlier_mean:.6f} -> "
            f"length={later_length} mean={later_mean:.6f}"
        )
    assert means[-1] < means[0], f"no net improvement from shortest to longest context: {means}"


# ---------------------------------------------------------------------------
# 6. substrate_bytes ACCOUNTING
# ---------------------------------------------------------------------------


def test_substrate_bytes_matches_independent_hand_count():
    dim = _DIM
    ttt = TTTLinear(dim=dim, inner_lr=_INNER_LR, seed=_BASE_SEED)
    breakdown = ttt.substrate_bytes_breakdown()
    print("SUBSTRATE BYTES BREAKDOWN:", breakdown)

    # Independent hand count, deliberately NOT calling anything in
    # src/models/ttt.py: 4 persistent tensors (theta_K, theta_Q, theta_V, W),
    # each dim x dim float32 elements, IEEE 754 float32 = 4 bytes/element.
    bytes_per_tensor = dim * dim * _BYTES_PER_FLOAT32
    hand_count = 4 * bytes_per_tensor

    assert breakdown == {
        "theta_K": bytes_per_tensor,
        "theta_Q": bytes_per_tensor,
        "theta_V": bytes_per_tensor,
        "W": bytes_per_tensor,
    }
    assert ttt.substrate_bytes() == hand_count
    print(f"substrate_bytes()={ttt.substrate_bytes()}  independent hand_count={hand_count}")


# ---------------------------------------------------------------------------
# 7. DETERMINISM
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_everything():
    a = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_DETERMINISM_SEED_A)
    b = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_DETERMINISM_SEED_A)
    assert _hash_base_state(a) == _hash_base_state(b), "same seed produced different base weights"

    xs = torch.randn(_DETERMINISM_STEPS, _DIM, generator=torch.Generator().manual_seed(_DETERMINISM_X_SEED))
    out_a = a.forward_sequence(xs.clone())
    out_b = b.forward_sequence(xs.clone())
    assert torch.equal(out_a, out_b), "same seed + same input produced different outputs"
    assert torch.equal(a.inner_loss_trace, b.inner_loss_trace), "same seed + same input produced different loss traces"


def test_determinism_different_seed_differs():
    """Negative control (M9): agreement is only evidence if it is hard to produce by accident."""
    a = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_DETERMINISM_SEED_A)
    b = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=_DETERMINISM_SEED_B)
    assert _hash_base_state(a) != _hash_base_state(b), "different seeds produced identical base weights"


# ---------------------------------------------------------------------------
# EDGE CASES (implementation/10_code_style.md's ladder; not a numbered
# acceptance test but required by the operating contract for every function)
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_int_dim_and_seed():
    with pytest.raises(TypeError):
        TTTLinear(dim=4.0, inner_lr=0.1, seed=0)
    with pytest.raises(TypeError):
        TTTLinear(dim=True, inner_lr=0.1, seed=0)  # bool is a subclass of int; rejected anyway
    with pytest.raises(TypeError):
        TTTLinear(dim=4, inner_lr=0.1, seed=1.5)


def test_constructor_rejects_non_positive_dim_seed_lr():
    with pytest.raises(ValueError):
        TTTLinear(dim=0, inner_lr=0.1, seed=0)
    with pytest.raises(ValueError):
        TTTLinear(dim=4, inner_lr=0.1, seed=-1)
    with pytest.raises(ValueError):
        TTTLinear(dim=4, inner_lr=0.0, seed=0)
    with pytest.raises(ValueError):
        TTTLinear(dim=4, inner_lr=-0.1, seed=0)
    with pytest.raises(ValueError):
        TTTLinear(dim=4, inner_lr=float("nan"), seed=0)


def test_constructor_accepts_boundary_dim_one():
    ttt = TTTLinear(dim=1, inner_lr=0.1, seed=0)
    out = ttt.forward_sequence(torch.randn(3, 1))
    assert out.shape == (3, 1)


def test_forward_sequence_rejects_non_float_dtype():
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=0)
    with pytest.raises(TypeError):
        ttt.forward_sequence(torch.zeros(3, _DIM, dtype=torch.long))


def test_forward_sequence_rejects_wrong_rank_and_wrong_dim():
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=0)
    with pytest.raises(ValueError):
        ttt.forward_sequence(torch.randn(_DIM))  # 1D, missing the seq axis
    with pytest.raises(ValueError):
        ttt.forward_sequence(torch.randn(3, _DIM + 1))  # feature dim mismatch


def test_forward_sequence_empty_sequence_is_a_no_op():
    ttt = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=0)
    before_hashes = _hash_base_state(ttt)
    out = ttt.forward_sequence(torch.zeros(0, _DIM))
    assert out.shape == (0, _DIM)
    assert ttt.inner_loss_trace.shape == (0,)
    assert _hash_base_state(ttt) == before_hashes

    # An empty call must not disturb a subsequent real call's result relative
    # to never having made the empty call at all.
    twin = TTTLinear(dim=_DIM, inner_lr=_INNER_LR, seed=0)
    xs = torch.randn(5, _DIM, generator=torch.Generator().manual_seed(11))
    assert torch.equal(ttt.forward_sequence(xs.clone()), twin.forward_sequence(xs.clone()))
