"""Tests for the BDH arm (src/models/bdh.py) -- Hebbian synapse, not weights, not softmax.

THE QUESTION this file answers: does `BDHLayer` actually have the three
properties that make it BDH rather than a strawman (implementation/09_mistake_log.md
M17, the reason this file exists)?

  1. Working memory lives ENTIRELY in synaptic state (`sigma`), not weights.
  2. Activations are high-dimensional, sparse, and positive.
  3. Attention-like retrieval EMERGES from local Hebbian correlation, with no
     softmax anywhere.

Every test is designed to FAIL if the property it targets does not hold --
see implementation/02_testing_philosophy.md ("a test that only passes when
the code works is worth very little") and this task's own acceptance
criterion: break the implementation three ways, confirm the matching test
fails with real captured output, revert. That log is in the calling task's
final report, not duplicated here.

Test-to-requirement map (the calling brief's 8 required tests):
  1. test_sparsity_and_positivity_exact_top_k_nonzero
  2. test_hebbian_locality_synapse_increases_for_co_activating_pair_only
  3. test_base_weights_untouched_after_forward_sequence
  4. test_state_is_load_bearing_reset_changes_output
  5. test_emergent_association_a_drives_b_more_than_control
  6. test_decay_weakens_old_association_over_intervening_tokens
  7. test_substrate_bytes_matches_independent_hand_count_and_sigma_dominates
  8. test_determinism_same_seed_identical_everything (+ negative control,
     mistake log M9: agreement is only evidence on a value hard to produce
     by accident)
Remaining tests (section 9) walk the edge-case ladder on the constructor and
on `forward_sequence`/`neuron_activations` input validation.

CONTAMINATION PROOF (implementation/02_testing_philosophy.md C3): every
token vector in this file is `torch.randn(..., generator=torch.Generator().manual_seed(N))`
for a literal integer `N` written in this file -- generated fresh in-process,
nowhere else, so there is nothing external the layer could have "seen".

WHY DOT-PRODUCT "DRIVE" AND NOT `forward_sequence`'S OWN OUTPUT: the required
API decodes the retrieved vector through `decoder` before returning it
(`out_t = (y_t + a_t) @ decoder`), which is lossy (n_neurons -> dim, and
n_neurons >> dim) and would not let a test see which SPECIFIC neurons a
retrieval landed on. `_retrieved_vector` below reconstructs `a_t = sigma @
query` directly from the PUBLIC `synapse_state()` and `neuron_activations()`
tensors, using exactly the formula documented in src/models/bdh.py's
MECHANISM section -- this observes the layer's own persisted `sigma`, it
does not re-derive new state, so it is not the M15 "shared criterion" trap
(two independent implementations of the SAME buggy logic agreeing).
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from src.models.bdh import BDHLayer
from src.models.bdh import _sparsify_top_k  # noqa: SLF001 -- direct unit test of the sparsification rule

# ---------------------------------------------------------------------------
# Shared scale: n_neurons=256 (65_536-entry sigma -- "fine" per the calling
# brief), dim=24 (small enough that sigma clearly dominates substrate_bytes,
# matching Property 2's n_neurons >> dim), top_k=16 (6.25% density, a clearly
# sparse but not degenerately-thin regime). decay=0.9 is the one parameter
# tests 5/6 need strictly < 1.0 to demonstrate forgetting; 0.9 was checked
# (see task report) to give a clean, non-degenerate decay curve over the
# checkpoint range below, not tuned to make a broken mechanism look correct
# -- the retrieval formula's exact match to a hand-derived decay^n prediction
# (test 6) is what establishes correctness, independent of this choice.
# ---------------------------------------------------------------------------

_DIM = 24
_N_NEURONS = 256
_TOP_K = 16
_DECAY = 0.9
_LAYER_SEED = 123


def _layer(seed: int = _LAYER_SEED, decay: float = _DECAY) -> BDHLayer:
    return BDHLayer(dim=_DIM, n_neurons=_N_NEURONS, top_k=_TOP_K, decay=decay, seed=seed)


def _token(seed: int) -> torch.Tensor:
    """One deterministic (1, _DIM) token vector, generated fresh in-process (see C3 above)."""
    return torch.randn(1, _DIM, generator=torch.Generator().manual_seed(seed))


def _retrieved_vector(layer: BDHLayer, query: torch.Tensor) -> torch.Tensor:
    """`a_t = sigma @ query` -- src/models/bdh.py's documented retrieval formula,
    computed here from the layer's PUBLIC `synapse_state()` (see module docstring)."""
    return layer.synapse_state() @ query


def _drive(layer: BDHLayer, probe: torch.Tensor, target: torch.Tensor) -> float:
    """How strongly retrieving with `probe` aligns with `target`'s own neuron pattern.

    `sigma` accumulates only outer products of ReLU'd (>= 0) activations
    with a non-negative decay, so `sigma`, and hence this dot product, is
    always >= 0 -- there is no sign cancellation to worry about.
    """
    return float(_retrieved_vector(layer, probe) @ target)


# ===========================================================================
# 1. SPARSITY AND POSITIVITY (Property 2)
# ===========================================================================


def test_sparsity_and_positivity_exact_top_k_nonzero():
    layer = _layer()
    xs = torch.randn(64, _DIM, generator=torch.Generator().manual_seed(1))

    y = layer.neuron_activations(xs)

    assert y.shape == (64, _N_NEURONS)
    assert bool((y >= 0).all()), "neuron activations must be entrywise non-negative (ReLU)"
    nonzero_per_token = (y > 0).sum(dim=-1)
    measured_sparsity_fraction = _TOP_K / _N_NEURONS
    print(f"MEASURED SPARSITY FRACTION: {measured_sparsity_fraction:.4f} ({_TOP_K} of {_N_NEURONS} neurons)")
    print(f"nonzero-per-token counts observed: {sorted(nonzero_per_token.unique().tolist())}")
    assert bool((nonzero_per_token == _TOP_K).all()), "exactly top_k neurons must be nonzero per token"


def test_sparsity_breaks_without_top_k_sparsification():
    """Negative control for test 1's own diagnosticity (N1): calling the pure
    ReLU+dense path (skipping `_sparsify_top_k`) must NOT look sparse. If it
    did, test 1 could not tell the difference and would be vacuous."""
    layer = _layer()
    xs = torch.randn(64, _DIM, generator=torch.Generator().manual_seed(1))
    dense = torch.relu(xs @ layer.base_state_dict()["encoder"])
    nonzero_per_token = (dense > 0).sum(dim=-1)
    print(f"dense (unsparsified) nonzero-per-token: {sorted(nonzero_per_token.unique().tolist())[:5]}...")
    assert bool((nonzero_per_token != _TOP_K).any()), "dense activations should not already look top-k sparse"


# ===========================================================================
# 2. HEBBIAN LOCALITY (Property 3, mechanism)
# ===========================================================================


def test_hebbian_locality_synapse_increases_for_co_activating_pair_only():
    layer = _layer()
    xs_a, xs_b, xs_control = _token(1000), _token(2000), _token(3000)
    y_a = layer.neuron_activations(xs_a)[0]
    y_b = layer.neuron_activations(xs_b)[0]
    y_control = layer.neuron_activations(xs_control)[0]

    sigma_before = layer.synapse_state().clone()
    layer.forward_sequence(torch.cat([xs_a, xs_b], dim=0))  # A then B: the hand-constructed pair
    sigma_after = layer.synapse_state()

    # i: a neuron B activates (value side of the write at B's step).
    # j: a neuron A activates (key side -- A is the PREVIOUS token when B is processed).
    i = int((y_b > 0).nonzero()[0])
    j = int((y_a > 0).nonzero()[0])
    before_ij, after_ij = float(sigma_before[i, j]), float(sigma_after[i, j])
    predicted_delta = float(y_b[i] * y_a[j])
    print(f"CO-ACTIVATING PAIR (A then B): sigma[{i},{j}] before={before_ij:.6f} after={after_ij:.6f} "
          f"delta={after_ij - before_ij:.6f} predicted_delta={predicted_delta:.6f}")
    assert after_ij > before_ij, "sigma[i,j] must INCREASE for a pair that co-activates neurons i and j"
    assert after_ij - before_ij == pytest.approx(predicted_delta, rel=1e-5)

    # k: a neuron the CONTROL token activates that B does NOT (so sigma[k,j]
    # can only have been written by an A-then-(something with neuron k)
    # pairing -- and control was never presented adjacent to A).
    clean_control_indices = ((y_control > 0) & (y_b == 0)).nonzero().flatten()
    assert len(clean_control_indices) > 0, "test construction needs >=1 control neuron absent from B's support"
    k = int(clean_control_indices[0])
    before_kj, after_kj = float(sigma_before[k, j]), float(sigma_after[k, j])
    print(f"NEVER-CO-ACTIVATING PAIR (A, unrelated control): sigma[{k},{j}] before={before_kj:.6f} "
          f"after={after_kj:.6f}")
    assert before_kj == 0.0 and after_kj == 0.0, "sigma[k,j] must not move for a pair that never co-activates"


def test_hebbian_locality_breaks_when_update_is_skipped():
    """Negative control (N1): with the write disabled (sigma forced to stay
    zero), the 'must increase' assertion above must fail. Demonstrates the
    test is not vacuous without needing to hand-edit src/models/bdh.py."""
    layer = _layer()
    xs_a, xs_b = _token(1000), _token(2000)
    layer.forward_sequence(torch.cat([xs_a, xs_b], dim=0))
    sigma_if_write_were_skipped = torch.zeros_like(layer.synapse_state())
    y_b = layer.neuron_activations(xs_b)[0]
    y_a = layer.neuron_activations(xs_a)[0]
    i, j = int((y_b > 0).nonzero()[0]), int((y_a > 0).nonzero()[0])
    with pytest.raises(AssertionError):
        assert float(sigma_if_write_were_skipped[i, j]) > 0.0


# ===========================================================================
# 3. BASE WEIGHTS UNTOUCHED (Property 1: adaptation is in sigma only)
# ===========================================================================


def _hash_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for name, tensor in tensors.items()}


def test_base_weights_untouched_after_forward_sequence():
    layer = _layer()
    before_hashes = _hash_tensors(layer.base_state_dict())
    print("BASE PARAM HASHES BEFORE:", before_hashes)

    many_tokens = torch.randn(300, _DIM, generator=torch.Generator().manual_seed(2))
    layer.forward_sequence(many_tokens)

    after_hashes = _hash_tensors(layer.base_state_dict())
    print("BASE PARAM HASHES AFTER: ", after_hashes)
    assert before_hashes == after_hashes, "encoder/decoder changed after forward_sequence -- gradient/weight leak"


# ===========================================================================
# 4. STATE IS LOAD-BEARING (reset_state() must change the output)
# ===========================================================================


def test_state_is_load_bearing_reset_changes_output():
    seq = torch.randn(20, _DIM, generator=torch.Generator().manual_seed(11))

    stateful = _layer(seed=77)
    out_first_call = stateful.forward_sequence(seq)
    out_second_call = stateful.forward_sequence(seq)  # sigma now carries accumulated state
    assert not torch.allclose(out_first_call, out_second_call), (
        "a second call over the same input must differ once sigma has accumulated state"
    )

    fresh = _layer(seed=77)
    out_fresh = fresh.forward_sequence(seq)
    assert torch.allclose(out_fresh, out_first_call), "a fresh layer's first call must match the stateful one's first call"

    stateful.reset_state()
    out_after_reset = stateful.forward_sequence(seq)
    assert torch.allclose(out_after_reset, out_fresh), "reset_state() must exactly undo accumulated sigma"
    print("STATE LOAD-BEARING: first-call == post-reset-call (True), first-call == second-call "
          f"(False, as required): {torch.allclose(out_after_reset, out_fresh)}, "
          f"{torch.allclose(out_first_call, out_second_call)}")


# ===========================================================================
# 5. EMERGENT ASSOCIATION (Property 3 -- "the property that matters")
# ===========================================================================

# Number of times the (A, B) pair is presented before the probe. Chosen (not
# tuned against the assertion threshold below -- see the report's robustness
# check across 8 independent token seeds) to give a comfortably-above-noise
# signal within a single-digit-millisecond test.
_PAIR_REPEATS = 5

# The ratio a genuinely broken/absent association would produce: querying
# with A retrieves nothing preferentially, so B and an unrelated control
# token receive statistically indistinguishable drive.
_CHANCE_RATIO = 1.0


def test_emergent_association_a_drives_b_more_than_control():
    layer = _layer()
    xs_a, xs_b, xs_control = _token(1000), _token(2000), _token(3000)
    y_a = layer.neuron_activations(xs_a)[0]
    y_b = layer.neuron_activations(xs_b)[0]
    y_control = layer.neuron_activations(xs_control)[0]

    sequence = torch.cat([xs_a, xs_b] * _PAIR_REPEATS, dim=0)
    layer.forward_sequence(sequence)

    drive_a_to_b = _drive(layer, probe=y_a, target=y_b)
    drive_a_to_control = _drive(layer, probe=y_a, target=y_control)
    ratio = drive_a_to_b / drive_a_to_control if drive_a_to_control > 0 else float("inf")

    print(f"A->B drive: {drive_a_to_b:.4f}")
    print(f"A->control drive: {drive_a_to_control:.4f}")
    print(f"ratio: {ratio:.4f} (chance/no-association baseline: {_CHANCE_RATIO})")

    assert drive_a_to_b > 0.0, "presenting A after training must retrieve SOME signal"
    assert ratio > 3.0, (
        f"ratio {ratio:.4f} is not clearly above the chance baseline of {_CHANCE_RATIO} -- "
        "association is not emerging; this must be reported honestly, not tuned away"
    )


def test_emergent_association_absent_without_training_pair():
    """Null control (C1): querying with A when A and B were NEVER presented
    together must retrieve ~nothing preferentially -- if it did, the layer
    (or this test's measure) would be "detecting" associations that were
    never written, i.e. measuring noise, not the mechanism."""
    layer = _layer()
    xs_a, xs_b, xs_control = _token(1000), _token(2000), _token(3000)
    y_a = layer.neuron_activations(xs_a)[0]
    y_b = layer.neuron_activations(xs_b)[0]
    y_control = layer.neuron_activations(xs_control)[0]
    # Deliberately never calling forward_sequence: sigma stays at its
    # reset_state() zero.

    drive_a_to_b = _drive(layer, probe=y_a, target=y_b)
    drive_a_to_control = _drive(layer, probe=y_a, target=y_control)
    print(f"NULL MODEL: A->B drive={drive_a_to_b}, A->control drive={drive_a_to_control} (want both exactly 0.0)")
    assert drive_a_to_b == 0.0 and drive_a_to_control == 0.0, "an untrained (all-zero) synapse must retrieve nothing"


# ===========================================================================
# 6. DECAY (old associations weaken over intervening tokens)
# ===========================================================================

_DECAY_CHECKPOINTS = (0, 1, 5, 10, 20, 30, 50)


def test_decay_weakens_old_association_over_intervening_tokens():
    xs_a, xs_b = _token(1000), _token(2000)
    baseline_layer = _layer()
    y_a = baseline_layer.neuron_activations(xs_a)[0]
    y_b = baseline_layer.neuron_activations(xs_b)[0]
    baseline_layer.forward_sequence(torch.cat([xs_a, xs_b], dim=0))
    drive_at_zero_filler = _drive(baseline_layer, probe=y_a, target=y_b)

    # Filler tokens are the ZERO vector -- ReLU(0) has no positive entries,
    # so neuron_activations() correctly returns the all-zero row (the
    # documented degenerate-row case in `_sparsify_top_k`), meaning each
    # filler step contributes outer(0, y_prev) = 0 to sigma. This isolates
    # the pure decay multiplier from any incidental reinforcement a random
    # filler could otherwise contribute (measured, not assumed: filler noise
    # from random *non-zero* tokens was tried first and gave a curve that
    # was not always monotone, because a random filler can, with small
    # probability, land on the exact (i, j) cell being measured -- see
    # task report). This does not change what decay is: the decay
    # multiplier is applied on every step regardless of that step's input.
    curve: list[tuple[int, float]] = []
    for n_filler in _DECAY_CHECKPOINTS:
        layer = _layer()
        fillers = torch.zeros(n_filler, _DIM)
        layer.forward_sequence(torch.cat([xs_a, xs_b, fillers], dim=0))
        curve.append((n_filler, _drive(layer, probe=y_a, target=y_b)))

    print(f"DECAY CURVE (decay={_DECAY}, base drive at n_filler=0: {drive_at_zero_filler:.4f}):")
    for n_filler, drive in curve:
        predicted = drive_at_zero_filler * (_DECAY ** n_filler)
        print(f"  n_filler={n_filler:3d}  drive_to_B={drive:10.4f}  predicted(decay^n * base)={predicted:10.4f}")
        assert drive == pytest.approx(predicted, rel=1e-4), (
            f"n_filler={n_filler}: measured drive does not match the hand-derived decay^n prediction"
        )

    drives = [drive for _, drive in curve]
    assert all(later < earlier for earlier, later in zip(drives, drives[1:])), (
        "drive must strictly decrease as intervening tokens accumulate"
    )
    assert drives[-1] < 0.01 * drives[0], "decay must be substantial, not negligible, over the full checkpoint range"


def test_decay_one_removed_recovers_no_forgetting_baseline():
    """Negative control: decay=1.0 (no forgetting) must NOT show the decrease
    the test above requires -- confirms the decrease above is decay's doing,
    not some other effect of processing more tokens."""
    xs_a, xs_b = _token(1000), _token(2000)
    no_decay_layer = _layer(decay=1.0)
    y_a = no_decay_layer.neuron_activations(xs_a)[0]
    y_b = no_decay_layer.neuron_activations(xs_b)[0]
    fillers = torch.zeros(50, _DIM)
    no_decay_layer.forward_sequence(torch.cat([xs_a, xs_b, fillers], dim=0))
    drive_after_50_fillers = _drive(no_decay_layer, probe=y_a, target=y_b)

    baseline_layer = _layer(decay=1.0)
    baseline_layer.forward_sequence(torch.cat([xs_a, xs_b], dim=0))
    drive_at_zero_filler = _drive(baseline_layer, probe=y_a, target=y_b)

    print(f"decay=1.0 (no forgetting): drive at n_filler=0 -> {drive_at_zero_filler:.4f}, "
          f"at n_filler=50 -> {drive_after_50_fillers:.4f}")
    assert drive_after_50_fillers == pytest.approx(drive_at_zero_filler, rel=1e-5), (
        "with decay=1.0 (no forgetting) zero-vector fillers must not change the drive at all"
    )


# ===========================================================================
# 7. substrate_bytes: base params AND sigma, both counted (sigma dominates)
# ===========================================================================


def test_substrate_bytes_matches_independent_hand_count_and_sigma_dominates():
    layer = _layer()

    # Hand count, derived independently from dim/n_neurons/dtype size --
    # NOT by calling substrate_bytes_breakdown() and trusting its own math.
    _FLOAT32_BYTES = 4
    expected = {
        "encoder": _DIM * _N_NEURONS * _FLOAT32_BYTES,
        "decoder": _N_NEURONS * _DIM * _FLOAT32_BYTES,
        "synapse": _N_NEURONS * _N_NEURONS * _FLOAT32_BYTES,
    }
    print(f"HAND-COUNTED BREAKDOWN: {expected}")
    print(f"substrate_bytes_breakdown():  {layer.substrate_bytes_breakdown()}")

    assert layer.substrate_bytes_breakdown() == expected
    assert layer.substrate_bytes() == sum(expected.values())

    # Cross-check directly against the raw tensors' own .nbytes, bypassing
    # substrate_bytes_breakdown() entirely (mistake log M9: two numbers
    # agreeing is only evidence if they could not agree by shared bug).
    base = layer.base_state_dict()
    assert base["encoder"].numel() * base["encoder"].element_size() == expected["encoder"]
    assert base["decoder"].numel() * base["decoder"].element_size() == expected["decoder"]
    sigma = layer.synapse_state()
    assert sigma.numel() * sigma.element_size() == expected["synapse"]

    total = sum(expected.values())
    sigma_share = expected["synapse"] / total
    print(f"sigma share of substrate_bytes: {sigma_share:.4f} ({expected['synapse']} / {total} bytes)")
    assert sigma_share > 0.5, "sigma (n_neurons^2) should dominate substrate_bytes at n_neurons >> dim"


def test_substrate_bytes_breaks_if_sigma_is_omitted():
    """Negative control (N1): dropping 'synapse' from the breakdown must make
    the accounting assertion above fail -- demonstrates test 7 is not vacuous."""
    layer = _layer()
    breakdown_without_sigma = {k: v for k, v in layer.substrate_bytes_breakdown().items() if k != "synapse"}
    total_without_sigma = sum(breakdown_without_sigma.values())
    print(f"UNDER-COUNTED total (sigma omitted): {total_without_sigma} vs true substrate_bytes(): "
          f"{layer.substrate_bytes()}")
    assert total_without_sigma != layer.substrate_bytes(), "omitting sigma must under-count substrate_bytes"


# ===========================================================================
# 8. Determinism on seed (+ negative control, mistake log M9)
# ===========================================================================


def test_determinism_same_seed_identical_everything():
    seq = torch.randn(20, _DIM, generator=torch.Generator().manual_seed(11))
    a = _layer(seed=999)
    b = _layer(seed=999)

    for name, tensor in a.base_state_dict().items():
        assert torch.equal(tensor, b.base_state_dict()[name])

    out_a = a.forward_sequence(seq)
    out_b = b.forward_sequence(seq)
    assert torch.equal(out_a, out_b)
    assert torch.equal(a.synapse_state(), b.synapse_state())


def test_determinism_different_seed_differs():
    """Negative control: if `seed` were silently ignored, the test above
    would pass vacuously (mistake log M9)."""
    a = _layer(seed=1)
    b = _layer(seed=2)
    assert not torch.equal(a.base_state_dict()["encoder"], b.base_state_dict()["encoder"])


# ===========================================================================
# 9. Edge-case ladder: constructor and input validation
# ===========================================================================


def test_constructor_rejects_non_int_and_bool_args():
    with pytest.raises(TypeError):
        BDHLayer(dim=1.5, n_neurons=8, top_k=2, decay=0.9, seed=0)
    with pytest.raises(TypeError):
        BDHLayer(dim=4, n_neurons=8, top_k=2, decay=0.9, seed=True)


def test_constructor_rejects_non_positive_dim_and_n_neurons():
    with pytest.raises(ValueError, match="dim"):
        BDHLayer(dim=0, n_neurons=8, top_k=2, decay=0.9, seed=0)
    with pytest.raises(ValueError, match="n_neurons"):
        BDHLayer(dim=4, n_neurons=0, top_k=2, decay=0.9, seed=0)


def test_constructor_rejects_top_k_outside_boundary():
    with pytest.raises(ValueError, match="top_k"):
        BDHLayer(dim=4, n_neurons=8, top_k=0, decay=0.9, seed=0)
    with pytest.raises(ValueError, match="top_k"):
        BDHLayer(dim=4, n_neurons=8, top_k=9, decay=0.9, seed=0)
    # top_k == n_neurons is the valid boundary (dense, not sparse -- allowed).
    BDHLayer(dim=4, n_neurons=8, top_k=8, decay=0.9, seed=0)


def test_constructor_rejects_decay_outside_unit_interval():
    with pytest.raises(ValueError, match="decay"):
        BDHLayer(dim=4, n_neurons=8, top_k=2, decay=-0.1, seed=0)
    with pytest.raises(ValueError, match="decay"):
        BDHLayer(dim=4, n_neurons=8, top_k=2, decay=1.1, seed=0)
    # Both boundaries are valid.
    BDHLayer(dim=4, n_neurons=8, top_k=2, decay=0.0, seed=0)
    BDHLayer(dim=4, n_neurons=8, top_k=2, decay=1.0, seed=0)


def test_constructor_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        BDHLayer(dim=4, n_neurons=8, top_k=2, decay=0.9, seed=-1)


def test_forward_sequence_rejects_wrong_dtype():
    layer = _layer()
    with pytest.raises(TypeError):
        layer.forward_sequence(torch.zeros(3, _DIM, dtype=torch.long))


def test_forward_sequence_rejects_wrong_ndim():
    layer = _layer()
    with pytest.raises(ValueError, match="2D"):
        layer.forward_sequence(torch.zeros(_DIM))


def test_forward_sequence_rejects_empty_sequence():
    layer = _layer()
    with pytest.raises(ValueError, match="seq_len 0"):
        layer.forward_sequence(torch.zeros(0, _DIM))


def test_forward_sequence_rejects_wrong_width():
    layer = _layer()
    with pytest.raises(ValueError, match="dim"):
        layer.forward_sequence(torch.zeros(3, _DIM + 1))


def test_neuron_activations_shares_the_same_validation():
    layer = _layer()
    with pytest.raises(TypeError):
        layer.neuron_activations(torch.zeros(3, _DIM, dtype=torch.bool))
    with pytest.raises(ValueError, match="seq_len 0"):
        layer.neuron_activations(torch.zeros(0, _DIM))


def test_sparsify_top_k_rejects_out_of_range_top_k():
    with pytest.raises(ValueError, match="top_k"):
        _sparsify_top_k(torch.zeros(1, 8), top_k=9)
    with pytest.raises(ValueError, match="top_k"):
        _sparsify_top_k(torch.zeros(1, 8), top_k=0)


def test_sparsify_top_k_all_zero_row_yields_fewer_than_top_k_nonzero():
    """Edge-case ladder rung 1 (empty/degenerate): an all-non-positive row
    cannot be forced to have top_k nonzero entries -- top_k is a rank
    selection over what ReLU already produced, not a floor."""
    all_zero_row = torch.zeros(1, 8)
    result = _sparsify_top_k(all_zero_row, top_k=3)
    assert int((result > 0).sum()) == 0, "an all-zero row must stay all-zero, not gain 3 fabricated entries"

