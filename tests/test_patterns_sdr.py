"""Tests for the pattern encoder (src/patterns/sdr.py) -- Components 1-2 of the
thesis ("Input -> Dynamic Pattern Encoder -> Pattern Space").

Read implementation/02_testing_philosophy.md before extending this file: a test
that only passes when the code works is worth very little. Every property test
here compares an ANALYTIC (exact, closed-form) prediction against an independent
MONTE CARLO measurement with a stated sampling-error model (implementation/09_mistake_log.md
M16: assert the exact expected value the construction implies, never a loose
floor). See the task report for the "deliberate breakage" verification log --
three targeted breaks, each confirmed to make a specific test FAIL with real
captured pytest output, then reverted -- which is the actual evidence these
tests work, not just that they pass.

Section numbering mirrors the eight ACCEPTANCE CRITERIA properties in the task
brief for this file (Vandermonde/overlap-set cardinality, expected overlap,
false-positive probability, capacity, noise robustness, union membership, the
semantic same-vs-different-archetype property, and determinism/sparsity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data.schema import ATTRIBUTES, Person
from src.data.structured import _archetype_supports, _value_lookup, generate_structured_people
from src.patterns.sdr import (
    SDR,
    N_CANONICAL,
    THETA_MATCH_CANONICAL,
    W_CANONICAL,
    _partition_sizes,
    bind,
    bundle,
    capacity_bits,
    capacity_patterns,
    encode_person,
    encode_token,
    expected_overlap,
    false_positive_membership_rate,
    false_positive_probability,
    matches,
    matching_probability_after_flip,
    noisy_copy,
    overlap,
    overlap_set_cardinality,
    sum_overlap_set_cardinalities,
    union,
    union_expected_size,
)

# All property tests operate at the module's own canonical operating point
# (Numenta's canonical SDR, n=2048/w=40) so results are directly comparable to
# the exact values given in the task brief for this file.
N, W = N_CANONICAL, W_CANONICAL


# ===========================================================================
# Shared helper: an empirical overlap sample, reused by properties 2 and 3b.
# ===========================================================================

# Sample size for the shared random-pair-overlap fixture. Large enough that
# property 2's sample standard error (~0.006 at this N; see the printed [property
# 2] line) is a small fraction of the mean it is testing (0.78), and that
# property 3b's smallest tested probability (theta=4, ~0.0069) still has an
# expected count in the hundreds (see [property 3b] printed expected_count).
_OVERLAP_SAMPLE_PAIRS = 20_000
_OVERLAP_SAMPLE_SEED = 1001


def _random_pair_overlaps(n: int, w: int, n_pairs: int, seed: int) -> np.ndarray:
    """overlap() of `n_pairs` independent random-token SDR pairs at (n, w).

    Each pair is drawn from two distinct, never-repeated token strings, so
    pairs are independent draws of two uniformly-random w-sparse SDRs (this is
    what "random SDR" means throughout this file: encode_token applied to an
    arbitrary, never-reused string -- see encode_token's docstring for why its
    hash-then-sample construction makes this a good proxy for a uniform random
    w-subset).
    """
    overlaps = np.empty(n_pairs, dtype=np.int64)
    for i in range(n_pairs):
        a = encode_token(f"pairA-{i}", n, w, seed)
        b = encode_token(f"pairB-{i}", n, w, seed)
        overlaps[i] = overlap(a, b)
    return overlaps


@pytest.fixture(scope="module")
def random_pair_overlaps() -> np.ndarray:
    return _random_pair_overlaps(N, W, _OVERLAP_SAMPLE_PAIRS, _OVERLAP_SAMPLE_SEED)


# ===========================================================================
# 0. SDR dataclass: structural validity, and the composition primitives
#    (overlap, matches, union, bind, bundle) that everything else is built on.
# ===========================================================================


def test_sdr_rejects_unsorted_active():
    with pytest.raises(ValueError):
        SDR(n=10, active=(3, 1, 2))


def test_sdr_rejects_duplicate_active():
    with pytest.raises(ValueError):
        SDR(n=10, active=(1, 1, 2))


def test_sdr_rejects_out_of_range_active():
    with pytest.raises(ValueError):
        SDR(n=10, active=(1, 2, 10))
    with pytest.raises(ValueError):
        SDR(n=10, active=(-1, 2, 3))


def test_sdr_rejects_non_int_active_element():
    with pytest.raises(TypeError):
        SDR(n=10, active=(1, 2.5))  # type: ignore[arg-type]


def test_sdr_rejects_bool_n():
    with pytest.raises(TypeError):
        SDR(n=True, active=())  # type: ignore[arg-type]


def test_sdr_empty_active_is_a_valid_boundary():
    empty = SDR(n=10, active=())
    assert len(empty.active) == 0


def test_sdr_fully_dense_active_is_a_valid_boundary():
    full = SDR(n=5, active=(0, 1, 2, 3, 4))
    assert len(full.active) == 5


def test_overlap_requires_matching_spaces():
    a, b = SDR(n=10, active=(1, 2, 3)), SDR(n=20, active=(1, 2, 3))
    with pytest.raises(ValueError):
        overlap(a, b)


def test_overlap_of_a_pattern_with_itself_is_full_sparsity():
    a = encode_token("self-overlap", N, W, seed=1)
    assert overlap(a, a) == W


def test_matches_boundary_theta():
    a = SDR(n=10, active=(0, 1, 2, 3, 4))
    b = SDR(n=10, active=(0, 1, 2, 5, 6))
    assert overlap(a, b) == 3
    assert matches(a, b, 3) is True
    assert matches(a, b, 4) is False


def test_matches_rejects_non_int_theta():
    a = SDR(n=10, active=(0,))
    with pytest.raises(TypeError):
        matches(a, a, 1.5)  # type: ignore[arg-type]


def test_union_of_zero_patterns_is_empty():
    assert union([]) == frozenset()


def test_union_requires_matching_spaces():
    a, b = SDR(n=10, active=(1,)), SDR(n=20, active=(1,))
    with pytest.raises(ValueError):
        union([a, b])


def test_union_of_identical_patterns_is_that_patterns_active_set():
    a = encode_token("union-one", N, W, seed=1)
    assert union([a, a]) == frozenset(a.active)


def test_bind_is_exactly_similarity_preserving():
    """The property the task API asks of bind(): EXACT, not approximate."""
    a1 = encode_token("bind-a1", N, W, seed=1)
    a2 = encode_token("bind-a2", N, W, seed=1)
    role = encode_token("bind-role", N, W, seed=1)
    assert overlap(bind(a1, role), bind(a2, role)) == overlap(a1, a2)


def test_bind_preserves_sparsity_and_space():
    a = encode_token("bind-sparsity", N, W, seed=1)
    role = encode_token("bind-role-2", N, W, seed=1)
    bound = bind(a, role)
    assert len(bound.active) == len(a.active)
    assert bound.n == a.n


def test_bind_requires_matching_spaces():
    a, b = SDR(n=10, active=(1,)), SDR(n=20, active=(1,))
    with pytest.raises(ValueError):
        bind(a, b)


def test_bind_is_not_generally_symmetric():
    a = encode_token("bind-x", N, W, seed=2)
    b = encode_token("bind-y", N, W, seed=2)
    assert bind(a, b).active != bind(b, a).active


def test_bundle_single_pattern_is_a_no_op():
    a = encode_token("bundle-solo", N, W, seed=1)
    assert bundle([a], W) == a


def test_bundle_requires_at_least_one_pattern():
    with pytest.raises(ValueError):
        bundle([], 5)


def test_bundle_rejects_mismatched_spaces():
    a, b = SDR(n=10, active=(1, 2)), SDR(n=20, active=(1, 2))
    with pytest.raises(ValueError):
        bundle([a, b], 2)


def test_bundle_rejects_insufficient_candidates():
    a, b = SDR(n=10, active=(0, 1)), SDR(n=10, active=(0, 1))
    with pytest.raises(ValueError):
        bundle([a, b], 5)  # union has only 2 distinct positions, need 5


def test_bundle_tie_break_is_deterministic_and_vote_ordered():
    # positions 2 and 4 appear in all three patterns (vote 3); 0, 1, 3 appear
    # once each (vote 1). Selecting w=3 must take {2, 4} plus the LOWEST-index
    # remaining candidate (0), by the documented tie-break rule.
    a, b, c = SDR(n=10, active=(0, 2, 4)), SDR(n=10, active=(1, 2, 4)), SDR(n=10, active=(2, 3, 4))
    result = bundle([a, b, c], 3)
    assert result.active == (0, 2, 4)
    assert bundle([a, b, c], 3) == result  # repeat call is bit-identical


# ===========================================================================
# 1. OVERLAP SET CARDINALITY: |Omega(n,w,b)| and the Vandermonde self-check.
# ===========================================================================


def test_overlap_set_cardinality_boundary_b_equals_w_is_one():
    # Only the reference w-subset itself shares all w elements with itself.
    assert overlap_set_cardinality(N, W, W) == 1


def test_overlap_set_cardinality_boundary_b_equals_zero():
    # b=0: choose all w active positions from the n-w positions the reference
    # does NOT use -- C(w,0)*C(n-w,w) = C(n-w,w).
    assert overlap_set_cardinality(N, W, 0) == math.comb(N - W, W)


def test_overlap_set_cardinality_rejects_b_out_of_range():
    with pytest.raises(ValueError):
        overlap_set_cardinality(N, W, W + 1)
    with pytest.raises(ValueError):
        overlap_set_cardinality(N, W, -1)


def test_vandermonde_identity_exact():
    """Property 1's acceptance criterion: sum over b=0..w of |Omega(n,w,b)|
    equals C(n,w) EXACTLY (integer equality, not approximate)."""
    total = sum_overlap_set_cardinalities(N, W)
    exact = math.comb(N, W)
    print(f"[property 1] sum_b |Omega(n,w,b)| = {total}")
    print(f"[property 1] C(n,w)               = {exact}")
    assert total == exact


# ===========================================================================
# 2. EXPECTED OVERLAP of two random SDRs = w^2/n. Analytic vs Monte Carlo.
# ===========================================================================


def test_expected_overlap_exact_closed_form_value():
    # 40^2/2048 = 1600/2048 = 25/32 = 0.78125 exactly (a terminating binary
    # fraction, so this is a safe EXACT float comparison, not approx).
    assert expected_overlap(N, W) == 0.78125


def test_expected_overlap_analytic_vs_monte_carlo(random_pair_overlaps: np.ndarray) -> None:
    analytic = expected_overlap(N, W)
    empirical_mean = float(random_pair_overlaps.mean())
    sample_std = float(random_pair_overlaps.std(ddof=1))
    n_pairs = len(random_pair_overlaps)
    sem = sample_std / math.sqrt(n_pairs)
    z = abs(empirical_mean - analytic) / sem
    print(
        f"[property 2] analytic={analytic:.5f} empirical={empirical_mean:.5f} "
        f"sample_std={sample_std:.4f} sem={sem:.5f} z={z:.2f} n_pairs={n_pairs}"
    )
    # Sampling-error model: overlap(a,b) for two independent random w-subsets
    # has finite variance; by the CLT the sample mean of n_pairs i.i.d. draws
    # is approximately Normal(analytic, sem^2). |z| < 5 keeps the false-failure
    # rate for a correct implementation below ~6e-7 (two-sided normal tail)
    # while still catching a real formula bug (e.g. a missing w^2 term would
    # move the mean by hundreds of sems, not a handful).
    assert z < 5.0, f"empirical mean is {z:.1f} sample-SEMs from analytic {analytic}"


# ===========================================================================
# 3. FALSE POSITIVE PROBABILITY at match threshold theta.
# ===========================================================================
#
# Two complementary checks, because the four externally-verified theta values
# (40, 20, 12, 8) have probabilities from 1e-7 down to 1e-85 -- far too small
# for ANY feasible sample size to observe directly (even theta=8 needs ~2e6
# samples to expect a single event, and theta=40/20/12 are astronomically
# smaller). So: (3a) reproduce those four exact values directly from the
# closed-form function (M16: exact value, not a bound); (3b) Monte Carlo test
# the SAME formula at smaller, actually-observable thetas, which is what
# certifies the formula itself -- not just these four memorised constants --
# is correctly implemented.

# Verified independently before this file was written (python3.11 one-liner
# over math.comb, cross-checked against the Vandermonde identity). Copied
# here literally from the task specification -- NOT imported from
# src.patterns.sdr's own internal constant, so this is a genuine, non-circular
# check against an externally-supplied expectation.
_VERIFIED_FALSE_POSITIVE_AT_THETA = {
    40: 4.216e-85,
    20: 2.491e-26,
    12: 1.980e-12,
    8: 4.975e-07,
}


def test_false_positive_probability_reproduces_verified_values():
    for theta, expected in _VERIFIED_FALSE_POSITIVE_AT_THETA.items():
        got = false_positive_probability(N, W, theta)
        rel_err = abs(got - expected) / expected
        print(f"[property 3a] theta={theta:3d} got={got:.4e} expected={expected:.4e} rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"theta={theta}: got {got}, expected {expected} (rel_err={rel_err})"


# Smaller thresholds where fp(n,w,theta) is large enough to sample directly
# (see report_false_positive_table-style printed [property 3b] expected_count).
_OBSERVABLE_THETAS = (1, 2, 3, 4)


def test_false_positive_probability_monte_carlo_at_observable_theta(
    random_pair_overlaps: np.ndarray,
) -> None:
    n_pairs = len(random_pair_overlaps)
    for theta in _OBSERVABLE_THETAS:
        analytic = false_positive_probability(N, W, theta)
        empirical_rate = float((random_pair_overlaps >= theta).mean())
        # Binomial sampling-error model: "does this pair's overlap reach
        # theta" is a Bernoulli(analytic) trial under the hypothesis being
        # tested; n_pairs i.i.d. trials give SEM = sqrt(p(1-p)/n_pairs).
        sem = math.sqrt(analytic * (1 - analytic) / n_pairs)
        z = abs(empirical_rate - analytic) / sem
        print(
            f"[property 3b] theta={theta} analytic={analytic:.5f} empirical={empirical_rate:.5f} "
            f"sem={sem:.5f} z={z:.2f} expected_count={analytic * n_pairs:.1f}"
        )
        assert z < 5.0, f"theta={theta}: empirical {empirical_rate} is {z:.1f} SEMs from analytic {analytic}"


def test_false_positive_probability_theta_beyond_w_is_zero():
    assert false_positive_probability(N, W, W + 1) == 0.0


def test_false_positive_probability_theta_at_or_below_zero_is_one():
    assert false_positive_probability(N, W, 0) == 1.0
    assert false_positive_probability(N, W, -5) == 1.0


# ===========================================================================
# 4. CAPACITY = C(n,w). Mistake M2: capacity in BITS is log2(C(n,w)), never
#    C(n,w) itself.
# ===========================================================================

# Computed independently before this file was written:
#   (lgamma(2049) - lgamma(41) - lgamma(2009)) / log(2)
# M16: assert the exact value, not a loose bound.
_EXACT_CAPACITY_BITS = 280.28792926074675


def test_capacity_patterns_has_85_digits():
    patterns = capacity_patterns(N, W)
    assert patterns == math.comb(N, W)
    assert len(str(patterns)) == 85, f"C({N},{W}) has {len(str(patterns))} digits, expected 85"


def test_capacity_bits_is_log2_not_the_raw_count():
    """The M2 lesson, asserted in code: capacity_bits is a small number of
    hundreds of bits; capacity_patterns is an 85-digit integer. Confusing them
    was mistake M2 (implementation/09_mistake_log.md)."""
    patterns = capacity_patterns(N, W)
    bits = capacity_bits(N, W)
    print(f"[property 4] capacity_patterns digits={len(str(patterns))} capacity_bits={bits:.6f}")
    assert bits == pytest.approx(_EXACT_CAPACITY_BITS, abs=1e-6)
    assert bits == pytest.approx(math.log2(float(patterns)), abs=1e-6)
    assert bits < 1_000 < 10**80 < patterns


def test_capacity_bits_boundary_w_equals_zero_and_w_equals_n():
    assert capacity_bits(N, 0) == 0.0  # C(n,0) = 1 pattern -> 0 bits of address space
    assert capacity_bits(N, N) == 0.0  # C(n,n) = 1 pattern -> 0 bits


# ===========================================================================
# 5. NOISE ROBUSTNESS: flip k active bits; matching at theta must reproduce
#    the analytically predicted (exact, degenerate) probability.
# ===========================================================================


@pytest.mark.parametrize("k", [0, 1, 10, 20, 39, 40])
def test_noisy_copy_overlap_is_exactly_w_minus_k(k: int):
    reference = encode_token("noise-reference", N, W, seed=4242)
    corrupted = noisy_copy(reference, k, seed=7)
    assert overlap(reference, corrupted) == W - k
    assert len(corrupted.active) == W


def test_noisy_copy_rejects_k_beyond_available_inactive_bits():
    small = encode_token("small-space", n=50, w=45, seed=1)  # only 5 inactive bits
    with pytest.raises(ValueError):
        noisy_copy(small, k=6, seed=1)


def test_noise_robustness_matches_analytic_probability_via_monte_carlo():
    theta = THETA_MATCH_CANONICAL
    reference = encode_token("noise-robustness-reference", N, W, seed=4242)
    n_trials = 200
    # Straddles the exact match/no-match boundary at k = W - theta.
    boundary_ks = (0, 1, W - theta - 1, W - theta, W - theta + 1, W - 1, W)
    for k in boundary_ks:
        predicted = matching_probability_after_flip(W, k, theta)
        observed = [
            matches(reference, noisy_copy(reference, k, seed=trial), theta) for trial in range(n_trials)
        ]
        empirical_rate = sum(observed) / n_trials
        print(f"[property 5] k={k:2d} theta={theta} predicted={predicted} MC_rate={empirical_rate} (n={n_trials})")
        # Degenerate probability (see matching_probability_after_flip's
        # docstring): the Monte Carlo rate must equal the prediction EXACTLY,
        # not merely approximately -- any deviation reveals a bug in
        # noisy_copy's supposedly-deterministic bit accounting.
        assert empirical_rate == predicted, (
            f"k={k}: Monte Carlo match rate {empirical_rate} != exact predicted {predicted}"
        )


# ===========================================================================
# 6. UNION MEMBERSHIP: expected union size and false-positive membership
#    rate, analytic vs Monte Carlo.
# ===========================================================================

_UNION_SIZE_TRIAL_M = 50
_UNION_SIZE_REPEATS = 300
_UNION_SIZE_SEED = 2024


def test_union_expected_size_analytic_vs_monte_carlo():
    m = _UNION_SIZE_TRIAL_M
    sizes = np.empty(_UNION_SIZE_REPEATS, dtype=np.int64)
    for r in range(_UNION_SIZE_REPEATS):
        patterns = [encode_token(f"union-size-{r}-{i}", N, W, _UNION_SIZE_SEED) for i in range(m)]
        sizes[r] = len(union(patterns))
    analytic = union_expected_size(N, W, m)
    empirical_mean = float(sizes.mean())
    sem = float(sizes.std(ddof=1)) / math.sqrt(_UNION_SIZE_REPEATS)
    z = abs(empirical_mean - analytic) / sem
    print(f"[property 6a] m={m} analytic={analytic:.3f} empirical={empirical_mean:.3f} sem={sem:.4f} z={z:.2f}")
    assert z < 5.0, f"empirical union size is {z:.1f} SEMs from analytic {analytic}"


def test_union_expected_size_boundary_m_zero_and_m_one():
    assert union_expected_size(N, W, 0) == 0.0
    assert union_expected_size(N, W, 1) == pytest.approx(float(W), abs=1e-9)


# m=130 puts false_positive_membership_rate in an observable ~4% range at
# (N, W) -- see the printed [property 6b] line for the actual value.
_MEMBERSHIP_UNION_M = 130
_MEMBERSHIP_QUERIES = 20_000
_MEMBERSHIP_SEED = 3030


def test_false_positive_membership_rate_analytic_vs_monte_carlo():
    stored = [encode_token(f"union-member-{i}", N, W, _MEMBERSHIP_SEED) for i in range(_MEMBERSHIP_UNION_M)]
    stored_union = union(stored)
    union_size = len(stored_union)
    # Uses the ACTUAL realised union size for this trial (not the theoretical
    # expected size), so the comparison isolates whether the false-positive
    # FORMULA is right, rather than conflating it with union-size sampling
    # noise (which test_union_expected_size_analytic_vs_monte_carlo already
    # covers separately).
    analytic = false_positive_membership_rate(N, W, union_size)

    false_positives = 0
    for i in range(_MEMBERSHIP_QUERIES):
        query = encode_token(f"union-query-{i}", N, W, _MEMBERSHIP_SEED + 1)
        if set(query.active).issubset(stored_union):
            false_positives += 1
    empirical_rate = false_positives / _MEMBERSHIP_QUERIES
    sem = math.sqrt(analytic * (1 - analytic) / _MEMBERSHIP_QUERIES)
    z = abs(empirical_rate - analytic) / sem
    print(
        f"[property 6b] m={_MEMBERSHIP_UNION_M} union_size={union_size} analytic={analytic:.5f} "
        f"empirical={empirical_rate:.5f} sem={sem:.5f} z={z:.2f} "
        f"false_positives={false_positives}/{_MEMBERSHIP_QUERIES}"
    )
    assert z < 5.0, f"empirical false-positive rate is {z:.1f} SEMs from analytic {analytic}"


# ===========================================================================
# 7. SEMANTIC PROPERTY: same-archetype people (Dataset B) must produce
#    patterns that overlap MORE than different-archetype people. This is the
#    single most important test in this file.
# ===========================================================================

_ARCHETYPE_K = 8
_ARCHETYPE_DEVIATION_BITS = 2
_ARCHETYPE_N_PEOPLE = 4_000
_ARCHETYPE_ENCODE_SEED = 55
_ARCHETYPE_PAIR_SAMPLES = 3_000


def _archetype_labels(people: list[Person], k: int, deviation_bits: int) -> list[int]:
    """Recover each person's latent archetype from an OBSERVED attribute
    value, using the same disjoint per-archetype supports
    src.data.structured drew from (mirrors tests/test_data_structured.py's
    own recovery helper). Only ATTRIBUTES[0] is used: disjoint supports mean
    every attribute recovers the same archetype for a given person
    (test_data_structured.py's own test 1b verifies that invariant; not
    re-verified here)."""
    attr = ATTRIBUTES[0]
    supports = _archetype_supports(attr, k, deviation_bits)
    lookup = _value_lookup(attr)
    value_to_k = {
        lookup(int(supports[kk, d])): kk for kk in range(k) for d in range(2**deviation_bits)
    }
    return [value_to_k[p.attributes[attr]] for p in people]


def test_same_archetype_patterns_overlap_more_than_different_archetype():
    """THE property the thesis needs: similar inputs -> overlapping patterns."""
    people = list(
        generate_structured_people(
            _ARCHETYPE_N_PEOPLE, seed=17, n_archetypes=_ARCHETYPE_K, deviation_bits=_ARCHETYPE_DEVIATION_BITS
        )
    )
    archetypes = _archetype_labels(people, _ARCHETYPE_K, _ARCHETYPE_DEVIATION_BITS)
    patterns = [encode_person(p, N, W, seed=_ARCHETYPE_ENCODE_SEED) for p in people]

    by_archetype: dict[int, list[int]] = {}
    for i, k in enumerate(archetypes):
        by_archetype.setdefault(k, []).append(i)

    rng = np.random.default_rng(999)
    archetype_keys = list(by_archetype)
    same_overlaps: list[int] = []
    while len(same_overlaps) < _ARCHETYPE_PAIR_SAMPLES:
        k = int(rng.choice(archetype_keys))
        members = by_archetype[k]
        if len(members) < 2:
            continue
        i, j = (int(idx) for idx in rng.choice(members, size=2, replace=False))
        same_overlaps.append(overlap(patterns[i], patterns[j]))

    diff_overlaps: list[int] = []
    while len(diff_overlaps) < _ARCHETYPE_PAIR_SAMPLES:
        k1, k2 = (int(idx) for idx in rng.choice(archetype_keys, size=2, replace=False))
        i = int(rng.choice(by_archetype[k1]))
        j = int(rng.choice(by_archetype[k2]))
        diff_overlaps.append(overlap(patterns[i], patterns[j]))

    same_mean = float(np.mean(same_overlaps))
    diff_mean = float(np.mean(diff_overlaps))
    separation = same_mean - diff_mean
    print(f"[property 7] same-archetype mean overlap = {same_mean:.3f} (n={len(same_overlaps)})")
    print(f"[property 7] diff-archetype mean overlap = {diff_mean:.3f} (n={len(diff_overlaps)})")
    print(f"[property 7] separation                  = {separation:.3f}")

    assert same_mean > diff_mean, (
        f"same-archetype overlap ({same_mean}) is not greater than different-archetype "
        f"overlap ({diff_mean}) -- the encoder is not reflecting semantic similarity"
    )
    # A weak effect could still be ">" by sampling noise alone; require the
    # separation to clear the COMBINED standard error by a wide margin, not
    # merely clear zero.
    combined_sem = math.sqrt(
        np.var(same_overlaps, ddof=1) / len(same_overlaps) + np.var(diff_overlaps, ddof=1) / len(diff_overlaps)
    )
    z = separation / combined_sem
    print(f"[property 7] separation z-score = {z:.1f} sigma")
    assert z > 10.0, f"separation is only {z:.1f} combined-SEMs; too weak to trust"

    # Cross-check against the analytic prediction derivable from
    # encode_person's disjoint block-partition construction and Dataset B's
    # archetype+deviation model: same-archetype people share a given
    # attribute's value with probability 2**-deviation_bits, in which case
    # that attribute's ENTIRE w_i-bit sub-pattern is identical (guaranteed,
    # not merely likely); different-archetype people never share a value
    # (disjoint supports by construction), so only chance-level overlap
    # (w_i^2/n_i) remains.
    n_sizes = _partition_sizes(N, len(ATTRIBUTES))
    w_sizes = _partition_sizes(W, len(ATTRIBUTES))
    p_match = 2.0**-_ARCHETYPE_DEVIATION_BITS
    predicted_same = sum(
        p_match * w_i + (1 - p_match) * (w_i**2 / n_i) for n_i, w_i in zip(n_sizes, w_sizes)
    )
    predicted_diff = sum((w_i**2) / n_i for n_i, w_i in zip(n_sizes, w_sizes))
    print(f"[property 7] analytic predicted same={predicted_same:.3f} diff={predicted_diff:.3f}")
    assert abs(same_mean - predicted_same) < 1.0, (
        f"empirical same-archetype mean {same_mean} far from analytic prediction {predicted_same}"
    )
    assert abs(diff_mean - predicted_diff) < 0.5, (
        f"empirical diff-archetype mean {diff_mean} far from analytic prediction {predicted_diff}"
    )


# ===========================================================================
# 8. ONE INPUT -> ONE PATTERN, deterministic; different inputs -> different
#    patterns; exactly w bits active, on every path.
# ===========================================================================


def test_same_input_and_seed_gives_bit_identical_sdr():
    a = encode_token("determinism-check", N, W, seed=5)
    b = encode_token("determinism-check", N, W, seed=5)
    assert a == b
    assert a.active == b.active


def test_different_tokens_give_different_sdrs():
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
    patterns = [encode_token(t, N, W, seed=6) for t in tokens]
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            assert patterns[i].active != patterns[j].active, f"{tokens[i]!r} and {tokens[j]!r} collided"


def test_different_seeds_give_different_sdrs_for_the_same_token():
    a = encode_token("same-token", N, W, seed=1)
    b = encode_token("same-token", N, W, seed=2)
    assert a.active != b.active


@pytest.mark.parametrize(
    "n,w",
    [(2048, 40), (100, 0), (100, 100), (1, 0), (1, 1), (500, 1), (2048, 2048), (0, 0)],
)
def test_encode_token_always_produces_exactly_w_active_bits(n: int, w: int):
    sdr = encode_token("some-token", n, w, seed=1)
    assert len(sdr.active) == w
    assert sdr.n == n


def test_encode_person_deterministic_same_seed():
    person = Person(person_id="Determinism Test-0002", attributes={a: f"v-{a}" for a in ATTRIBUTES})
    a = encode_person(person, N, W, seed=9)
    b = encode_person(person, N, W, seed=9)
    assert a == b


def test_encode_person_different_attribute_value_changes_the_pattern():
    base_attrs = {a: f"v-{a}" for a in ATTRIBUTES}
    p1 = Person(person_id="P-0001", attributes=dict(base_attrs))
    changed = dict(base_attrs)
    changed[ATTRIBUTES[0]] = "a-completely-different-value"
    p2 = Person(person_id="P-0002", attributes=changed)
    s1 = encode_person(p1, N, W, seed=9)
    s2 = encode_person(p2, N, W, seed=9)
    assert s1.active != s2.active


@pytest.mark.parametrize("n,w", [(2048, 40), (50, 10), (500, 25)])
def test_encode_person_always_produces_exactly_w_active_bits(n: int, w: int):
    person = Person(person_id="Test Person-0001", attributes={a: f"val-{a}" for a in ATTRIBUTES})
    sdr = encode_person(person, n, w, seed=1)
    assert len(sdr.active) == w
    assert sdr.n == n


def test_worked_example_first_ten_active_indices():
    """The concrete, eyeball-able example: reproducible from a fresh process."""
    sdr = encode_token("Kelmoran Vushiel-4417", N, W, seed=42)
    assert len(sdr.active) == W
    assert sdr.active[:10] == (8, 69, 112, 240, 244, 394, 428, 519, 543, 545)


# ===========================================================================
# 9. Input validation: type/range errors (edge-case ladder rungs 2, 3, 4, 5).
# ===========================================================================


@pytest.mark.parametrize(
    "n,w,seed,expected_exc",
    [
        (-1, 0, 0, ValueError),  # negative n
        (10, -1, 0, ValueError),  # negative w
        (10, 11, 0, ValueError),  # w > n
        (10.0, 1, 0, TypeError),  # non-int n
        (10, 1.0, 0, TypeError),  # non-int w
        (10, True, 0, TypeError),  # bool rejected as w
        (10, 1, 1.5, TypeError),  # non-int seed
    ],
)
def test_encode_token_rejects_invalid_input(n: object, w: object, seed: object, expected_exc: type[Exception]):
    with pytest.raises(expected_exc):
        encode_token("x", n, w, seed)  # type: ignore[arg-type]


def test_encode_token_rejects_non_str_token():
    with pytest.raises(TypeError):
        encode_token(123, 10, 1, 0)  # type: ignore[arg-type]


def test_encode_person_rejects_non_person():
    with pytest.raises(TypeError):
        encode_person("not a person", N, W, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "n,w,b,expected_exc",
    [(-1, 0, 0, ValueError), (10, 11, 0, ValueError), (10, 5, 6, ValueError), (10, 5, -1, ValueError)],
)
def test_overlap_set_cardinality_rejects_invalid_input(n: int, w: int, b: int, expected_exc: type[Exception]):
    with pytest.raises(expected_exc):
        overlap_set_cardinality(n, w, b)


def test_false_positive_probability_rejects_non_int_theta():
    with pytest.raises(TypeError):
        false_positive_probability(N, W, 1.5)  # type: ignore[arg-type]


def test_capacity_functions_reject_w_greater_than_n():
    with pytest.raises(ValueError):
        capacity_patterns(10, 11)
    with pytest.raises(ValueError):
        capacity_bits(10, 11)


def test_expected_overlap_rejects_n_zero():
    with pytest.raises(ValueError):
        expected_overlap(0, 0)


def test_union_expected_size_rejects_negative_m():
    with pytest.raises(ValueError):
        union_expected_size(N, W, -1)


def test_false_positive_membership_rate_rejects_union_size_out_of_range():
    with pytest.raises(ValueError):
        false_positive_membership_rate(N, W, N + 1)
    with pytest.raises(ValueError):
        false_positive_membership_rate(N, W, -1)


def test_false_positive_membership_rate_union_size_below_w_is_zero():
    # math.comb(union_size, w) == 0 when union_size < w: cannot fit w active
    # positions inside a union smaller than w -- exactly zero, not "small".
    assert false_positive_membership_rate(N, W, W - 1) == 0.0


@pytest.mark.parametrize(
    "k,seed,expected_exc",
    [(-1, 0, ValueError), (41, 0, ValueError), (1.5, 0, TypeError), (1, 1.5, TypeError)],
)
def test_noisy_copy_rejects_invalid_input(k: object, seed: object, expected_exc: type[Exception]):
    reference = encode_token("noisy-copy-validation", N, W, seed=1)
    with pytest.raises(expected_exc):
        noisy_copy(reference, k, seed)  # type: ignore[arg-type]
