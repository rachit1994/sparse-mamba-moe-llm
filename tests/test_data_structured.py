"""Tests for Dataset B, the structured biography generator (src/data/structured.py).

Read implementation/02_testing_philosophy.md before extending this file: a test
that only passes when the code works is worth very little -- every test here is
designed to FAIL if the specific defect it targets is present. See the task
report for the "deliberate breakage" verification log (real captured pytest
output for each of the three required breaks -- overlapping supports, drawing
attributes independently, ignoring the seed -- plus a byte-identical revert
confirmation) -- that log is the actual evidence these tests work, not just
that they pass.

Sample sizes are chosen to keep the whole file well under the ~90s / 4-core
budget while still being large enough for the entropy and mutual-information
estimators to be trustworthy (see each test's own tolerance justification).
"""

from __future__ import annotations

import itertools
import math
from typing import Final

import numpy as np
import pytest

from src.data import generate as g
from src.data.generate import generate_people, render_probe_qa
from src.data.schema import ATTRIBUTES, Person
from src.data.structured import (
    _archetype_supports,
    _validate_structured_config,
    _value_lookup,
    compression_ratio,
    generate_structured_people,
    structured_entropy_bits,
)
from src.data.tokenizer import build_tokenizer

# A config reused by the tests that do not care about its specific shape (only
# that it is a small, valid, non-degenerate one). needed = 4 * 2**2 = 16 <= 100
# (schema.ATTRIBUTE_CARDINALITY["major"], the tightest attribute).
_DEFAULT_K: Final[int] = 4
_DEFAULT_DEV_BITS: Final[int] = 2


# ===========================================================================
# 1. Empirical entropy from a large sample matches the analytic formula.
# ===========================================================================
#
# A plug-in entropy estimator over the raw 5-attribute JOINT distribution
# would need K * (2**deviation_bits)**5 samples-per-outcome to be trustworthy
# -- intractable at realistic K/deviation_bits in a container-scale test. So
# this test instead re-derives the SAME decomposition the module docstring's
# proof uses (H(archetype) + sum_j H(value_j | archetype)), each term over a
# small number of bins (K or 2**deviation_bits), estimated directly from
# OBSERVED attribute strings -- not from re-reading the generator's internal
# RNG draws. Recovering "which archetype" from an observed value is itself
# only valid because supports are disjoint; see test 1b below, which checks
# that recovery is consistent across attributes (i.e. checks the invariant
# the proof needs, not just the arithmetic downstream of it).

_ENTROPY_N: Final[int] = 100_000
_ENTROPY_K: Final[int] = 8
_ENTROPY_DEV_BITS: Final[int] = 2  # D=4; needed=32 <= 100 (major)

# Plug-in Shannon entropy is NEGATIVELY biased (Miller 1955): sampling noise
# makes an empirical distribution look sharper than the true one. Standard
# first-order correction: add back (bins-1)/(2N ln2). This is the same
# correction tests/test_data_generate.py applies to its MI estimator (MI is a
# difference of entropies, so the same bias enters there too).
def _plugin_entropy_bits(labels: np.ndarray, n_bins: int) -> float:
    """Bias-corrected plug-in Shannon entropy of `labels`, in bits."""
    n = labels.size
    counts = np.bincount(labels, minlength=n_bins).astype(np.float64)
    counts = counts[counts > 0]
    p = counts / n
    h_naive = float(-(p * np.log2(p)).sum())
    bias_correction = (n_bins - 1) / (2 * n * math.log(2))
    return h_naive + bias_correction


def _archetype_and_slot_labels(
    people: list[Person], attribute: str, n_archetypes: int, deviation_bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recover each person's archetype k and within-archetype slot d for
    `attribute` from the OBSERVED value string alone, using the same disjoint
    supports `generate_structured_people` drew from. This is what lets the
    entropy test check the analytic formula against values the generator
    actually emitted, rather than re-deriving the numbers the code already
    computed internally.
    """
    supports = _archetype_supports(attribute, n_archetypes, deviation_bits)  # (K, D) domain idx
    lookup = _value_lookup(attribute)
    value_to_kd = {
        lookup(int(supports[k, d])): (k, d)
        for k in range(n_archetypes)
        for d in range(2**deviation_bits)
    }
    archetypes = np.empty(len(people), dtype=np.int64)
    slots = np.empty(len(people), dtype=np.int64)
    for i, person in enumerate(people):
        k, d = value_to_kd[person.attributes[attribute]]
        archetypes[i] = k
        slots[i] = d
    return archetypes, slots


def test_archetype_recovery_agrees_across_attributes() -> None:
    """Test 1b: every attribute must recover the SAME archetype for the same
    person. This is the on-data check of "H(archetype | attributes) = 0" that
    the entropy proof depends on -- disjoint supports guarantee it
    mathematically, but this checks the actual generated data, not just the
    construction argument.
    """
    people = list(
        generate_structured_people(
            5_000, seed=111, n_archetypes=_ENTROPY_K, deviation_bits=_ENTROPY_DEV_BITS
        )
    )
    reference_archetypes, _ = _archetype_and_slot_labels(
        people, ATTRIBUTES[0], _ENTROPY_K, _ENTROPY_DEV_BITS
    )
    for attr in ATTRIBUTES[1:]:
        archetypes, _ = _archetype_and_slot_labels(people, attr, _ENTROPY_K, _ENTROPY_DEV_BITS)
        assert np.array_equal(archetypes, reference_archetypes), (
            f"{attr!r} recovered a different archetype than {ATTRIBUTES[0]!r} "
            f"for at least one person -- archetype supports are not disjoint"
        )


# Tolerance justification (stated, not asserted-away): at N=100,000 with
# K=8 and per-archetype-conditional bins of size D=4, the Miller-Madow
# correction removes the dominant first-order bias; residual error is
# sampling variance of a handful of small-alphabet plug-in estimates, which
# empirically lands two orders of magnitude below this bound (see printed
# `abs_err` in the test's own output). 0.05 bits is ~0.4% of the 13.0-bit
# analytic value for this config -- tight enough to catch a real defect (a
# support-overlap bug moves this by whole bits; see the deliberate-breakage
# log in the task report) while not being so tight that ordinary sampling
# noise trips it.
_ENTROPY_TOLERANCE_BITS: Final[float] = 0.05


def test_empirical_entropy_matches_analytic_formula() -> None:
    people = list(
        generate_structured_people(
            _ENTROPY_N, seed=321, n_archetypes=_ENTROPY_K, deviation_bits=_ENTROPY_DEV_BITS
        )
    )
    archetype_labels, _ = _archetype_and_slot_labels(
        people, ATTRIBUTES[0], _ENTROPY_K, _ENTROPY_DEV_BITS
    )

    d = 2**_ENTROPY_DEV_BITS
    h_archetype_hat = _plugin_entropy_bits(archetype_labels, _ENTROPY_K)
    h_deviation_hat = 0.0
    for attr in ATTRIBUTES:
        _, slots = _archetype_and_slot_labels(people, attr, _ENTROPY_K, _ENTROPY_DEV_BITS)
        for k in range(_ENTROPY_K):
            mask = archetype_labels == k
            weight = float(mask.sum()) / _ENTROPY_N
            h_deviation_hat += weight * _plugin_entropy_bits(slots[mask], d)

    empirical_bits = h_archetype_hat + h_deviation_hat
    analytic_bits = structured_entropy_bits(1, _ENTROPY_K, _ENTROPY_DEV_BITS)
    abs_err = abs(empirical_bits - analytic_bits)
    print(
        f"[entropy] analytic={analytic_bits:.6f} empirical={empirical_bits:.6f} "
        f"abs_err={abs_err:.6f} (tolerance {_ENTROPY_TOLERANCE_BITS})"
    )
    assert abs_err < _ENTROPY_TOLERANCE_BITS, f"empirical vs analytic entropy: abs_err={abs_err}"


# ===========================================================================
# 2. Mutual information: > 0 for Dataset B, ~0 for Dataset A. Side by side.
# ===========================================================================
#
# Reproduces (does not import) tests/test_data_generate.py's bias-corrected
# plug-in MI estimator, so this file's controls are self-contained and do not
# depend on another test file's internals.

_MI_N: Final[int] = 20_000
_MI_K: Final[int] = 6
_MI_DEV_BITS: Final[int] = 2  # D=4; needed=24 <= 100 (major)
_MI_BINS: Final[int] = 10

_VALUE_TO_INDEX: Final[dict[str, dict[str, int]]] = {
    attr: {v: i for i, v in enumerate(g._attribute_value_vocab(attr))}  # noqa: SLF001
    for attr in ("birth_city", "university", "major", "employer")
}


def _value_bucket(attribute: str, value: str, k: int = _MI_BINS) -> int:
    """Map a raw attribute value to one of `k` bins, same recipe as
    tests/test_data_generate.py's `_bucket_index`."""
    if attribute == "birth_date":
        year = int(value.rsplit(" ", 1)[-1])
        return year % k
    return _VALUE_TO_INDEX[attribute][value] % k


def _mi_bits(x_bins: np.ndarray, y_bins: np.ndarray, k: int = _MI_BINS) -> float:
    """Bias-corrected (Miller-Madow) plug-in mutual information, in bits."""
    n = len(x_bins)
    joint = np.zeros((k, k))
    for x, y in zip(x_bins, y_bins):
        joint[x, y] += 1
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
    mi_naive = float(terms.sum())
    bias = (k - 1) * (k - 1) / (2 * n * math.log(2))
    return max(0.0, mi_naive - bias)


# Dataset A threshold: identical value and reasoning to
# tests/test_data_generate.py's _MI_THRESHOLD_BITS (finite-sample bias at
# N=20,000, k=10 is ~0.0029 bits; 0.02 is a ~7x margin).
_MI_THRESHOLD_A_BITS: Final[float] = 0.02
# Dataset B threshold: with K=6 archetypes, knowing one attribute's bucket
# should carry real information about another's via the shared archetype.
# Set well below the smallest MI actually measured (see printed output) and
# well above the Dataset-A threshold, so the two datasets are unambiguously
# distinguished by the same estimator.
_MI_THRESHOLD_B_BITS: Final[float] = 0.05


def test_mutual_information_structured_positive_flat_near_zero() -> None:
    flat_people = list(generate_people(_MI_N, seed=555))
    structured_people = list(
        generate_structured_people(_MI_N, seed=556, n_archetypes=_MI_K, deviation_bits=_MI_DEV_BITS)
    )

    flat_bins = {
        attr: np.array([_value_bucket(attr, p.attributes[attr]) for p in flat_people])
        for attr in ATTRIBUTES
    }
    structured_bins = {
        attr: np.array([_value_bucket(attr, p.attributes[attr]) for p in structured_people])
        for attr in ATTRIBUTES
    }

    for a, b in itertools.combinations(ATTRIBUTES, 2):
        mi_a = _mi_bits(flat_bins[a], flat_bins[b])
        mi_b = _mi_bits(structured_bins[a], structured_bins[b])
        print(f"[MI] {a} x {b}: dataset_A={mi_a:.5f} bits   dataset_B={mi_b:.5f} bits")
        assert mi_a < _MI_THRESHOLD_A_BITS, f"Dataset A {a}x{b}: MI={mi_a} >= near-zero threshold"
        assert mi_b > _MI_THRESHOLD_B_BITS, f"Dataset B {a}x{b}: MI={mi_b} <= minimum-signal threshold"


# ===========================================================================
# 3. compression_ratio > 1 for every offered config; rises as K falls.
# ===========================================================================

# deviation_bits fixed at 1 (D=2); K ascending so "rises as K falls" is
# checked as strictly-descending ratios walking this list in order.
# needed=K*2 <= 100 (major) for every entry, including the K=50 boundary.
_COMPRESSION_K_SWEEP: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 50)
_COMPRESSION_K_SWEEP_DEV_BITS: Final[int] = 1

# A second, independent config family varying deviation_bits at fixed K, to
# cover "every offered config" beyond a single sweep dimension.
_COMPRESSION_DEV_BITS_SWEEP: Final[tuple[int, ...]] = (0, 1, 2, 3)
_COMPRESSION_DEV_BITS_SWEEP_K: Final[int] = 4


def test_compression_ratio_exceeds_one_for_every_offered_config() -> None:
    for k in _COMPRESSION_K_SWEEP:
        ratio = compression_ratio(k, _COMPRESSION_K_SWEEP_DEV_BITS)
        print(f"[compression] K={k} deviation_bits={_COMPRESSION_K_SWEEP_DEV_BITS} -> {ratio:.4f}")
        assert ratio > 1.0, f"K={k}: compression_ratio={ratio} <= 1"
    for dev_bits in _COMPRESSION_DEV_BITS_SWEEP:
        ratio = compression_ratio(_COMPRESSION_DEV_BITS_SWEEP_K, dev_bits)
        print(f"[compression] K={_COMPRESSION_DEV_BITS_SWEEP_K} deviation_bits={dev_bits} -> {ratio:.4f}")
        assert ratio > 1.0, f"deviation_bits={dev_bits}: compression_ratio={ratio} <= 1"


def test_compression_ratio_rises_as_k_falls() -> None:
    ratios = [compression_ratio(k, _COMPRESSION_K_SWEEP_DEV_BITS) for k in _COMPRESSION_K_SWEEP]
    # _COMPRESSION_K_SWEEP is ascending in K, so "ratio rises as K falls"
    # means ratios must be strictly DESCENDING when walked in that order.
    for ratio_at_lower_k, ratio_at_higher_k in zip(ratios, ratios[1:]):
        assert ratio_at_lower_k > ratio_at_higher_k, (
            f"compression_ratio did not rise as K fell: {ratios} for K in {_COMPRESSION_K_SWEEP}"
        )


def test_compression_ratio_degenerate_config_raises() -> None:
    with pytest.raises(ValueError):
        compression_ratio(1, 0)


# ===========================================================================
# 4. Determinism on seed.
# ===========================================================================


def test_determinism_same_seed_identical_output() -> None:
    a = list(
        generate_structured_people(2_000, seed=42, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    b = list(
        generate_structured_people(2_000, seed=42, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    assert a == b


def test_different_seed_different_output() -> None:
    """Companion sanity check: seed is not silently ignored in the other
    direction either -- different seeds must not coincidentally agree."""
    a = list(
        generate_structured_people(2_000, seed=1, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    b = list(
        generate_structured_people(2_000, seed=2, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    assert a != b


# ===========================================================================
# 5. Person IDs unique.
# ===========================================================================


def test_person_ids_unique() -> None:
    people = list(
        generate_structured_people(5_000, seed=987, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    ids = [p.person_id for p in people]
    assert len(set(ids)) == len(ids)


# ===========================================================================
# 6. A Dataset B Person passes through tokenizer.encode_qa without error.
# ===========================================================================


def test_structured_person_encodes_via_tokenizer_encode_qa() -> None:
    tokenizer = build_tokenizer()
    people = list(
        generate_structured_people(50, seed=13, n_archetypes=_DEFAULT_K, deviation_bits=_DEFAULT_DEV_BITS)
    )
    assert people, "fixture produced zero people; test would be vacuous"
    checked = 0
    for person in people:
        for attr in ATTRIBUTES:
            question, answer = render_probe_qa(person, attr, 0)
            ids, mask = tokenizer.encode_qa(question, answer)
            assert len(ids) == len(mask)
            assert any(mask), f"no answer tokens masked for {attr!r}"
            checked += 1
    assert checked == len(people) * len(ATTRIBUTES)


# ===========================================================================
# 7. Input validation: type/range/feasibility errors (edge-case ladder).
# ===========================================================================


@pytest.mark.parametrize(
    "n_people,seed,n_archetypes,deviation_bits,expected_exc",
    [
        (-1, 0, 4, 1, ValueError),  # negative n_people
        (10, -1, 4, 1, ValueError),  # negative seed
        (10, 0, 0, 1, ValueError),  # n_archetypes < 1
        (10, 0, 4, -1, ValueError),  # deviation_bits < 0
        (10, 0, 51, 1, ValueError),  # infeasible: 51*2=102 > 100 (major)
        (10.0, 0, 4, 1, TypeError),  # non-int n_people
        (10, 0, 4, True, TypeError),  # bool is not accepted as an int here
    ],
)
def test_generate_structured_people_rejects_invalid_input(
    n_people: object, seed: object, n_archetypes: object, deviation_bits: object, expected_exc: type[Exception]
) -> None:
    with pytest.raises(expected_exc):
        generate_structured_people(n_people, seed, n_archetypes, deviation_bits)  # type: ignore[arg-type]


def test_n_archetypes_equals_one_is_valid_boundary() -> None:
    """K=1 is a valid boundary: log2(1)=0, so entropy comes entirely from
    deviations. Not the useful case (no archetype signal to discover), but
    must not raise."""
    people = list(generate_structured_people(10, seed=0, n_archetypes=1, deviation_bits=2))
    assert len(people) == 10
    assert structured_entropy_bits(10, 1, 2) == 10 * (0.0 + len(ATTRIBUTES) * 2)


def test_deviation_bits_zero_is_valid_boundary() -> None:
    """deviation_bits=0 is a valid boundary: D=1, every person of a given
    archetype gets exactly that archetype's single preferred value for every
    attribute (zero within-archetype variety)."""
    people = list(generate_structured_people(20, seed=0, n_archetypes=4, deviation_bits=0))
    assert len(people) == 20
    archetypes, _ = _archetype_and_slot_labels(people, ATTRIBUTES[0], 4, 0)
    for attr in ATTRIBUTES[1:]:
        archetypes_j, _ = _archetype_and_slot_labels(people, attr, 4, 0)
        assert np.array_equal(archetypes_j, archetypes)


def test_zero_people_is_empty_and_zero_entropy() -> None:
    assert list(generate_structured_people(0, seed=0, n_archetypes=4, deviation_bits=2)) == []
    assert structured_entropy_bits(0, 4, 2) == 0.0


def test_validate_structured_config_names_offending_attribute() -> None:
    # needed = 101 * 2 = 202: exceeds "major" (cardinality 100, the tightest
    # attribute) but not "university" (300), which is checked first in
    # ATTRIBUTE_CARDINALITY's iteration order -- so this specifically
    # exercises that the error message names the truly-offending attribute,
    # not merely the first one checked.
    with pytest.raises(ValueError, match="major"):
        _validate_structured_config(101, 1)
