"""THE QUESTION this file answers: given an input, does the pattern-encoder form a
sparse binary PATTERN with the exact properties the SDR literature specifies -- and
does semantic similarity of the input show up as pattern OVERLAP?

This is the foundation of the whole thesis (implementation/08_thesis_alignment.md):
"Input -> Dynamic Pattern Encoder -> Pattern Space", where a concept IS a stable
sparse activation pattern. Every downstream mechanism (interaction, composition,
memory) presupposes patterns are formed correctly. THIS FILE HAS NO TRAINING LOOP
AND NO torch.nn.Module: an SDR encoder is a deterministic function from input to a
sparse binary vector, not a learned layer.

HOW TO REVIEW THIS FILE
    Section 1  CONSTANTS        -- every input, with provenance. Check these against
                                    initial_research/02_math_corrections.md and stop;
                                    if the constants are right, only the maths below
                                    can be wrong.
    Section 2  PURE FUNCTIONS   -- the SDR type plus one quantity per function, no
                                    printing, no module state. Each independently
                                    checkable.
    Section 3  SELF-CHECKS      -- identities this file depends on (Vandermonde,
                                    the four externally-verified false-positive
                                    values, the M2 count-vs-bits distinction, exact
                                    similarity preservation), asserted before any
                                    report output.
    Section 4  REPORT           -- formatting only, calls Section 2.

Run: python3.11 -m src.patterns.sdr     (exit 0 = all self-checks passed)

Grounding for the SDR formalism used throughout (overlap-set cardinality, expected
overlap, false-positive-at-threshold, capacity): Ahmad, S. & Hawkins, J. (2015),
"Properties of Sparse Distributed Representations and their Application to
Hierarchical Temporal Memory", and this repo's own from-scratch re-derivation in
initial_research/02_math_corrections.md and initial_research/verify_math.py.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb, lgamma, log, log2
from typing import Final

import numpy as np

from src.data.schema import ATTRIBUTES, Person

__all__ = [
    "SDR",
    "bind",
    "bundle",
    "capacity_bits",
    "capacity_patterns",
    "encode_person",
    "encode_token",
    "expected_overlap",
    "false_positive_membership_rate",
    "false_positive_probability",
    "matches",
    "matching_probability_after_flip",
    "noisy_copy",
    "overlap",
    "overlap_set_cardinality",
    "sum_overlap_set_cardinalities",
    "union",
    "union_expected_size",
]

# =============================================================================
# SECTION 1 -- CONSTANTS
# Every input to this file appears here, with provenance. Nothing is derived in
# this section.
# =============================================================================

#: Numenta's canonical SDR width. Source: initial_research/02_math_corrections.md
#: line 43, "Numenta's canonical SDR, C(2048, 40) ~= 10^84.4" (citing Ahmad &
#: Hawkins 2015). Used as the default operating point for every self-check and
#: worked example below; callers of encode_token/encode_person choose their own
#: (n, w) -- nothing in Section 2 hardcodes this value.
N_CANONICAL: Final[int] = 2048

#: Numenta's canonical SDR sparsity (40/2048 ~= 2% active). Same source as
#: N_CANONICAL.
W_CANONICAL: Final[int] = 40

#: False-positive probabilities at (N_CANONICAL, W_CANONICAL), computed
#: independently before this file was written (python3.11 one-liner over
#: math.comb, cross-checked against the Vandermonde identity -- see the task
#: brief for this module). run_self_checks() reproduces these from this file's
#: own false_positive_probability() to _FALSE_POSITIVE_REL_TOL.
_VERIFIED_FALSE_POSITIVE_AT_THETA: Final[dict[int, float]] = {
    40: 4.216e-85,
    20: 2.491e-26,
    12: 1.980e-12,
    8: 4.975e-07,
}

#: Relative tolerance for comparing this file's false_positive_probability()
#: against _VERIFIED_FALSE_POSITIVE_AT_THETA. The reference values are stated to
#: 4 significant figures, so 1e-3 comfortably separates "matches the reference to
#: its own precision" from "silently wrong" (e.g. dropping the C(n-w,w-b) factor,
#: per this module's task-specified ACCEPTANCE break (c)) without tripping on the
#: reference's own rounding.
_FALSE_POSITIVE_REL_TOL: Final[float] = 1e-3

#: Half of W_CANONICAL. Not an external citation -- derived and justified
#: in-file: at (N_CANONICAL, W_CANONICAL) this gives false_positive_probability
#: ~= 2.5e-26 (see _VERIFIED_FALSE_POSITIVE_AT_THETA[20]), so two unrelated
#: random patterns essentially never appear to match by chance, while still
#: tolerating up to W_CANONICAL - THETA_MATCH_CANONICAL = 20 corrupted active
#: bits (see matching_probability_after_flip) before a genuine match is lost.
#: Used only as this file's own report/self-check operating point.
THETA_MATCH_CANONICAL: Final[int] = 20

#: ln(2), the single named nats/bits-style conversion boundary for capacity_bits
#: (mirrors src/metrics/bits.py's _NATS_TO_BITS convention: one named constant,
#: never an inlined `/log(2)` scattered across the file).
_LN2: Final[float] = log(2.0)


# =============================================================================
# SECTION 2 -- PURE FUNCTIONS
# One quantity per function. No printing, no module state. Each is checkable in
# isolation.
# =============================================================================


@dataclass(frozen=True)
class SDR:
    """A sparse distributed representation: `n` bits, with `active` marking which are on.

    `active` is the sole source of truth for sparsity: there is no stored "w"
    field (none is in the required API), so len(active) IS the pattern's
    sparsity. The encoders below (encode_token, encode_person) are what
    guarantee len(active) == a caller-requested w; this type only enforces that
    whatever active set it is given is well-formed, since overlap()/bind()/
    bundle() below rely on `active` being sorted and duplicate-free for correct,
    efficient comparison -- a malformed active tuple would silently corrupt
    every downstream overlap count.

    Attributes:
        n: total bits in the representational space. Must be >= 0.
        active: indices of the "on" bits: strictly ascending, no duplicates,
            each in [0, n).

    Raises:
        TypeError: if n or any element of active is not an int (bools are
            rejected -- bool is a subclass of int in Python and is never a
            legitimate index or bit count).
        ValueError: if n < 0, active contains a duplicate, active is not
            strictly ascending, or any index falls outside [0, n).
    """

    n: int
    active: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int):
            raise TypeError(f"SDR.n must be an int, got {type(self.n).__name__}")
        if self.n < 0:
            raise ValueError(f"SDR.n must be non-negative, got {self.n}")
        if not isinstance(self.active, tuple):
            raise TypeError(f"SDR.active must be a tuple, got {type(self.active).__name__}")
        for idx in self.active:
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise TypeError(f"SDR.active must contain only ints, found {idx!r}")
        if len(set(self.active)) != len(self.active):
            raise ValueError(f"SDR.active contains a duplicate index: {self.active}")
        if list(self.active) != sorted(self.active):
            raise ValueError(f"SDR.active must be sorted ascending, got {self.active}")
        if self.active and (self.active[0] < 0 or self.active[-1] >= self.n):
            raise ValueError(
                f"SDR.active indices must lie in [0, {self.n}); got range "
                f"[{self.active[0]}, {self.active[-1]}]"
            )


def _validate_n_w(n: int, w: int) -> None:
    """Shared validation for every function taking an (n, w) SDR-space pair.

    Raises:
        TypeError: if n or w is not an int (bools rejected).
        ValueError: if n < 0 or w is outside [0, n].
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if isinstance(w, bool) or not isinstance(w, int):
        raise TypeError(f"w must be an int, got {type(w).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not (0 <= w <= n):
        raise ValueError(f"w must be in [0, {n}], got {w}")


def encode_token(token: str, n: int, w: int, seed: int) -> SDR:
    """Deterministically encode `token` as a w-sparse SDR over n bits.

    A pure hash-then-sample function: `token` and `seed` are combined through
    SHA-256 (never Python's built-in hash(), which is salted per-process unless
    PYTHONHASHSEED is fixed -- implementation/02_testing_philosophy.md section 4)
    into a 64-bit integer that seeds a numpy PCG64 generator, which draws w
    DISTINCT positions from [0, n) without replacement. Same (token, n, w, seed)
    always produces the bit-identical SDR; this is the one encoding primitive
    every other encoder in this file is built from.

    Args:
        token: the string to encode. Any str, including "".
        n: total representational bits. Non-negative int.
        w: active bits to draw. Int with 0 <= w <= n.
        seed: determinism seed, mixed into the hash alongside `token` so
            different seeds encode the same token to unrelated patterns. Any int.

    Returns:
        An SDR with exactly `n` and exactly `w` active bits (len(active) == w,
        always -- required by property 8 in this module's test suite).

    Raises:
        TypeError: if token is not a str, or n/w/seed is not an int.
        ValueError: if n < 0, or w is outside [0, n].
    """
    if not isinstance(token, str):
        raise TypeError(f"token must be a str, got {type(token).__name__}")
    _validate_n_w(n, w)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], "big")
    drawn = np.random.default_rng(rng_seed).choice(n, size=w, replace=False)
    return SDR(n=n, active=tuple(sorted(int(i) for i in drawn)))


def _partition_sizes(total: int, parts: int) -> tuple[int, ...]:
    """Split `total` into `parts` non-negative ints summing exactly to `total`,
    as evenly as possible: the first `total % parts` parts get one extra unit.

    Used by encode_person to divide both n and w across ATTRIBUTES so each
    attribute's sub-pattern lives in its own disjoint block (see
    encode_person's docstring for why disjointness is what turns a shared
    attribute value into a GUARANTEED overlap contribution).

    Raises:
        ValueError: if parts <= 0.
    """
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    base, remainder = divmod(total, parts)
    return tuple(base + (1 if i < remainder else 0) for i in range(parts))


def encode_person(person: Person, n: int, w: int, seed: int) -> SDR:
    """Deterministically encode `person` as a w-sparse SDR over n bits.

    Composition: partitions the n-bit space into len(ATTRIBUTES) disjoint
    blocks (_partition_sizes), encodes each "attr=value" string into its OWN
    block via encode_token, and takes the union() of the offset sub-patterns.
    Blocks are disjoint by construction, so the union has exactly w active
    bits with no tie-breaking or truncation (contrast bundle(), for the
    general non-disjoint case).

    This is what produces this module's semantic property (property 7,
    tests/test_patterns_sdr.py): encode_token is a pure function of its
    "attr=value" string, so two people sharing attribute VALUE V for
    attribute `attr` get IDENTICAL bits in that attribute's block -- a
    guaranteed w_i-bit overlap contribution, not merely a likely one. People
    differing on every attribute contribute only chance-level overlap
    (~w_i^2/n_i per block; see expected_overlap). Verified empirically against
    src.data.structured's archetypes, where same-archetype people share a
    given attribute's value with probability 2**-deviation_bits, by
    construction of that module's disjoint per-archetype supports.

    Args:
        person: the individual to encode (src.data.schema.Person -- this
            file's only dependency on the dataset layer).
        n: total representational bits. Non-negative int.
        w: active bits in the result. Int with 0 <= w <= n.
        seed: determinism seed, passed through to every per-attribute
            encode_token call.

    Returns:
        An SDR with exactly `n` and exactly `w` active bits.

    Raises:
        TypeError: if person is not a Person, or n/w/seed is not an int.
        ValueError: if n < 0 or w is outside [0, n].
    """
    if not isinstance(person, Person):
        raise TypeError(f"person must be a Person, got {type(person).__name__}")
    _validate_n_w(n, w)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    n_sizes = _partition_sizes(n, len(ATTRIBUTES))
    w_sizes = _partition_sizes(w, len(ATTRIBUTES))
    block_starts: list[int] = []
    acc = 0
    for size in n_sizes:
        block_starts.append(acc)
        acc += size

    sub_patterns: list[SDR] = []
    for i, attr in enumerate(ATTRIBUTES):
        # Defence in depth, not user-facing validation: w_sizes[i] <= n_sizes[i]
        # is a PROVEN invariant here, not merely a likely one. Both sequences
        # come from _partition_sizes(total, J) with the SAME J and the SAME
        # "first (total % J) blocks get +1" rule; since w <= n is already
        # enforced by _validate_n_w above, floor(w/J) <= floor(n/J), and when
        # the floors are equal, w <= n forces w % J <= n % J too -- so the +1
        # bonus can never push a w-block past its same-index n-block. Kept as
        # an assert (an internal invariant, not a caller mistake) so a future
        # change to how blocks are sized fails loudly instead of silently
        # corrupting encode_person's output.
        assert w_sizes[i] <= n_sizes[i], (
            f"internal invariant violated: attribute {attr!r} got w_sizes[{i}]="
            f"{w_sizes[i]} > n_sizes[{i}]={n_sizes[i]} despite w={w} <= n={n}"
        )
        token = f"{attr}={person.attributes[attr]}"
        sub = encode_token(token, n_sizes[i], w_sizes[i], seed)
        offset_active = tuple(sorted(block_starts[i] + idx for idx in sub.active))
        sub_patterns.append(SDR(n=n, active=offset_active))

    person_active = union(sub_patterns)
    assert len(person_active) == w, (
        f"encode_person: union of disjoint attribute blocks produced "
        f"{len(person_active)} active bits, expected exactly w={w} -- blocks are "
        f"supposed to be non-overlapping by construction; this indicates a logic "
        f"bug in the block partitioning above, not a legitimate encoding"
    )
    return SDR(n=n, active=tuple(sorted(person_active)))


def overlap(a: SDR, b: SDR) -> int:
    """|active(a) intersect active(b)|, the number of bits both patterns share.

    This is the SDR literature's fundamental similarity measure (Ahmad &
    Hawkins 2015): raw bit overlap, not Euclidean/cosine distance, since SDR
    bits are binary and equally weighted by construction.

    Raises:
        ValueError: if a.n != b.n (overlap is only meaningful within one
            shared n-bit space).
    """
    if a.n != b.n:
        raise ValueError(f"overlap requires matching spaces, got a.n={a.n} b.n={b.n}")
    return len(set(a.active) & set(b.active))


def matches(a: SDR, b: SDR, theta: int) -> bool:
    """Whether `a` and `b` overlap enough to be considered the same pattern.

    Args:
        theta: minimum overlap required. Values outside [0, w] are
            well-defined: theta <= 0 always matches, theta >
            min(len(a.active), len(b.active)) never does.

    Raises:
        TypeError: if theta is not an int.
        ValueError: if a.n != b.n (delegated to overlap()).
    """
    if isinstance(theta, bool) or not isinstance(theta, int):
        raise TypeError(f"theta must be an int, got {type(theta).__name__}")
    return overlap(a, b) >= theta


def union(patterns: Sequence[SDR]) -> frozenset[int]:
    """The set of every bit position active in ANY of `patterns`.

    The SDR "storage by superposition" primitive (Ahmad & Hawkins 2015):
    OR-ing w-sparse patterns together lets one n-bit register represent
    membership in M patterns at once, at the cost of a rising false-positive
    membership rate as the union fills (see false_positive_membership_rate,
    union_expected_size).

    Args:
        patterns: any number of SDRs, including zero.

    Returns:
        frozenset of active indices in the shared n-bit space. Empty if
        `patterns` is empty.

    Raises:
        ValueError: if patterns is non-empty and its elements do not all
            share the same n (positions from different spaces are not
            comparable).
    """
    if not patterns:
        return frozenset()
    n0 = patterns[0].n
    for p in patterns:
        if p.n != n0:
            raise ValueError(f"union requires matching spaces, got n={n0} and n={p.n}")
    return frozenset(idx for p in patterns for idx in p.active)


def bind(a: SDR, b: SDR) -> SDR:
    """Compose `a` with `b`, preserving `a`'s similarity structure exactly.

    A deterministic pseudo-random permutation of [0, n) is derived from `b`
    (its active tuple hashed via SHA-256, mirroring encode_token's own
    hash-to-seed step) and applied to `a`'s active indices. Because a
    permutation is a bijection, this is EXACTLY similarity-preserving for a
    fixed "role" b: for any a1, a2,

        overlap(bind(a1, b), bind(a2, b)) == overlap(a1, a2)

    exactly, not approximately -- a bijection applied identically to two sets
    preserves the size of their intersection. This is the "must be
    similarity-preserving" property the task's API asks of bind(), checked as
    an exact identity in run_self_checks() and tests/test_patterns_sdr.py
    (this project's M16 rule: where a construction implies an exact value,
    assert against that value, not a loose bound).

    `bind` is NOT symmetric in general: `b` plays a fixed "context"/"role",
    `a` the thing placed within it, so bind(a, b) != bind(b, a) in general.

    Raises:
        ValueError: if a.n != b.n.
    """
    if a.n != b.n:
        raise ValueError(f"bind requires matching spaces, got a.n={a.n} b.n={b.n}")
    digest = hashlib.sha256(f"{b.n}:{b.active}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], "big")
    permutation = np.random.default_rng(rng_seed).permutation(a.n)
    new_active = tuple(sorted(int(permutation[i]) for i in a.active))
    return SDR(n=a.n, active=new_active)


def bundle(patterns: Sequence[SDR], w: int) -> SDR:
    """Superpose `patterns` into ONE w-sparse SDR via weighted union pooling.

    Standard SDR bundling technique (contrast bind(), which keeps every
    operand recoverable; bundle() deliberately lossy-compresses): count how
    many input patterns have each bit position active (a "vote"), then keep
    the w positions with the most votes, breaking ties by lowest index. Tie
    breaking is deterministic, so bundle() is a pure function of its inputs.

    Args:
        patterns: the SDRs to combine. Must be non-empty (there is no n to
            derive a result from otherwise) and must all share one n.
        w: active bits in the result. Int with 0 <= w <= n.

    Returns:
        An SDR with len(active) == w, drawn from the highest-vote positions.

    Raises:
        ValueError: if patterns is empty, patterns do not all share one n, w
            is outside [0, n], or fewer than w distinct positions are active
            across all of `patterns`.
    """
    if not patterns:
        raise ValueError("bundle requires at least one pattern")
    n = patterns[0].n
    for p in patterns:
        if p.n != n:
            raise ValueError(f"bundle requires matching spaces, got n={n} and n={p.n}")
    _validate_n_w(n, w)

    votes = Counter(idx for p in patterns for idx in p.active)
    if len(votes) < w:
        raise ValueError(
            f"union of {len(patterns)} pattern(s) has only {len(votes)} distinct "
            f"active positions, fewer than the requested w={w}"
        )
    ranked = sorted(votes.items(), key=lambda vote: (-vote[1], vote[0]))
    return SDR(n=n, active=tuple(sorted(idx for idx, _ in ranked[:w])))


def overlap_set_cardinality(n: int, w: int, b: int) -> int:
    """|Omega(n,w,b)|: count of w-subsets of an n-set sharing exactly b
    elements with one FIXED reference w-subset.

    = C(w,b) * C(n-w,w-b): choose which b of the reference's w active bits
    are kept (C(w,b) ways), then choose the remaining w-b active bits from
    the n-w positions the reference does NOT use (C(n-w,w-b) ways;
    math.comb returns 0, not an error, when w-b > n-w, correctly encoding
    "impossible" with no extra branch here).

    Raises:
        ValueError: if n<0, w outside [0,n], or b outside [0,w].
    """
    _validate_n_w(n, w)
    if not (0 <= b <= w):
        raise ValueError(f"b must be in [0, {w}], got {b}")
    return comb(w, b) * comb(n - w, w - b)


def sum_overlap_set_cardinalities(n: int, w: int) -> int:
    """Sum of overlap_set_cardinality(n,w,b) over every b in [0,w].

    By the Vandermonde identity this must equal EXACTLY C(n,w): every
    w-subset of an n-set shares some b in [0,w] with the reference, so
    partitioning by b and summing recovers the whole space of w-subsets.
    Checked as an exact integer equality in run_self_checks() -- this is the
    identity this module's ACCEPTANCE break (c) (dropping the C(n-w,w-b)
    factor) is designed to fail.
    """
    return sum(overlap_set_cardinality(n, w, b) for b in range(w + 1))


def expected_overlap(n: int, w: int) -> float:
    """E[overlap(a,b)] for two INDEPENDENT uniformly-random w-sparse SDRs over n bits.

    Exact closed form: w^2/n. Derivation (linearity of expectation): fix a's
    active set; for each of a's w active positions, P(b is also active
    there) = w/n (b is a uniformly random w-subset), so the expected count
    of a's active positions also active in b is w * (w/n) = w^2/n.

    Raises:
        ValueError: if n<=0 or w outside [0,n].
    """
    _validate_n_w(n, w)
    if n == 0:
        raise ValueError("expected_overlap requires n > 0")
    return (w * w) / n


def false_positive_probability(n: int, w: int, theta: int) -> float:
    """P(overlap(a,b) >= theta) for two INDEPENDENT uniformly-random w-sparse
    SDRs over n bits -- the probability an unrelated pattern is mistaken for
    a match at threshold theta.

    = sum_{b=theta}^{w} overlap_set_cardinality(n,w,b) / C(n,w). theta
    outside [0,w] needs no special case: the summation range clamps
    naturally (empty for theta>w, giving 0.0; the full sum for theta<=0
    equals C(n,w) exactly by sum_overlap_set_cardinalities, giving 1.0).

    Raises:
        TypeError: if theta is not an int.
        ValueError: if n<0 or w outside [0,n].
    """
    _validate_n_w(n, w)
    if isinstance(theta, bool) or not isinstance(theta, int):
        raise TypeError(f"theta must be an int, got {type(theta).__name__}")
    lo = max(theta, 0)
    numerator = sum(overlap_set_cardinality(n, w, b) for b in range(lo, w + 1))
    return numerator / comb(n, w)


def capacity_patterns(n: int, w: int) -> int:
    """C(n,w): the COUNT of distinguishable w-sparse patterns over n bits.

    This is an address-space size, not a bit-capacity -- see capacity_bits.
    Conflating the two was mistake M2 (implementation/09_mistake_log.md):
    "10^241 patterns is 803 *bits*, not 10^241 bits".

    Raises:
        ValueError: if n<0 or w outside [0,n].
    """
    _validate_n_w(n, w)
    return comb(n, w)


def capacity_bits(n: int, w: int) -> float:
    """log2(C(n,w)): the actual information capacity of an n-bit, w-sparse code, in bits.

    NEVER read capacity_patterns(n,w) itself as a bit count -- see that
    function's docstring and M2. Computed via lgamma rather than
    log2(float(capacity_patterns(n,w))) so it stays correct even where
    capacity_patterns' huge integer would overflow float64 (e.g.
    C(100_000,1_000) ~ 10^8073, per initial_research/02_math_corrections.md's
    capacity table -- far beyond float64's ~10^308 range). Cross-checked
    against the direct (safe-range) computation in run_self_checks(),
    mirroring initial_research/verify_math.py's own lgamma cross-check.

    Raises:
        ValueError: if n<0 or w outside [0,n].
    """
    _validate_n_w(n, w)
    return (lgamma(n + 1) - lgamma(w + 1) - lgamma(n - w + 1)) / _LN2


def noisy_copy(sdr: SDR, k: int, seed: int) -> SDR:
    """Corrupt `sdr` by flipping exactly `k` of its active bits.

    `k` active positions (chosen by `seed`) are turned off and replaced by
    `k` NEW positions drawn from `sdr`'s previously-inactive pool (also
    chosen by `seed`). The result has the same len(active) as `sdr`
    (w-sparsity preserved) and, because replacements are drawn only from
    previously-inactive positions, EXACTLY `len(sdr.active) - k` of the
    original active bits survive -- deterministically, not in expectation.
    See matching_probability_after_flip, which depends on this being exact.

    Args:
        sdr: the pattern to corrupt.
        k: active bits to flip. Int with
            0 <= k <= min(len(sdr.active), sdr.n - len(sdr.active)).
        seed: determinism seed for which bits flip and where they land.

    Raises:
        TypeError: if k or seed is not an int.
        ValueError: if k is negative, exceeds len(sdr.active), or exceeds
            the number of inactive positions available to draw from.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    w = len(sdr.active)
    n_inactive = sdr.n - w
    if not (0 <= k <= w):
        raise ValueError(f"k must be in [0, {w}], got {k}")
    if k > n_inactive:
        raise ValueError(
            f"cannot draw {k} replacement bits from only {n_inactive} inactive positions"
        )
    if k == 0:
        return sdr

    rng = np.random.default_rng(seed)
    active_set = set(sdr.active)
    to_remove = {int(i) for i in rng.choice(np.array(sdr.active), size=k, replace=False)}
    inactive = np.array([i for i in range(sdr.n) if i not in active_set], dtype=np.int64)
    to_add = {int(i) for i in rng.choice(inactive, size=k, replace=False)}
    new_active = (active_set - to_remove) | to_add
    assert len(new_active) == w, (
        f"noisy_copy: expected {w} active bits after flipping {k}, got "
        f"{len(new_active)} -- to_remove/to_add overlapped unexpectedly"
    )
    return SDR(n=sdr.n, active=tuple(sorted(new_active)))


def matching_probability_after_flip(w: int, k: int, theta: int) -> float:
    """Exact probability that noisy_copy(sdr, k, seed) still matches its
    original `sdr` (of sparsity w) at threshold theta, for ANY seed.

    Because noisy_copy's surviving-original-bit count is deterministic
    (w - k, always -- see that function's docstring), this probability is
    degenerate: exactly 1.0 if w - k >= theta, exactly 0.0 otherwise. This
    is an identity, not an estimate -- this project's M16 rule
    (implementation/09_mistake_log.md) requires asserting against exactly
    this value. A Monte Carlo estimate landing strictly between 0 and 1
    would itself reveal a bug in noisy_copy (an accidental extra or missing
    surviving bit), which is what makes the Monte-Carlo-vs-analytic
    comparison in tests/test_patterns_sdr.py meaningful despite the answer
    being 0/1 rather than a "real" probability.

    Raises:
        TypeError: if k or theta is not an int.
        ValueError: if k is outside [0, w].
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if isinstance(theta, bool) or not isinstance(theta, int):
        raise TypeError(f"theta must be an int, got {type(theta).__name__}")
    if not (0 <= k <= w):
        raise ValueError(f"k must be in [0, {w}], got {k}")
    return 1.0 if (w - k) >= theta else 0.0


def union_expected_size(n: int, w: int, m: int) -> float:
    """E[|union of m INDEPENDENT uniformly-random w-sparse SDRs over n bits|].

    Exact closed form: n * (1 - (1 - w/n)^m). Derivation: fix a bit position
    i. Each of the m patterns independently leaves i inactive with
    probability 1 - w/n (i is inactive in a uniformly random w-subset with
    probability (n-w)/n). P(i inactive in ALL m) = (1-w/n)^m, so
    P(i active in >= 1) = 1-(1-w/n)^m; summing over all n positions
    (linearity of expectation) gives the formula.

    Raises:
        TypeError: if m is not an int.
        ValueError: if n<=0, w outside [0,n], or m<0.
    """
    _validate_n_w(n, w)
    if n == 0:
        raise ValueError("union_expected_size requires n > 0")
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError(f"m must be an int, got {type(m).__name__}")
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    return n * (1.0 - (1.0 - w / n) ** m)


def false_positive_membership_rate(n: int, w: int, union_size: int) -> float:
    """P(a freshly-drawn, independent random w-sparse SDR is falsely reported
    as "already in the union") for a union of exactly `union_size` distinct
    active positions.

    Exact closed form: C(union_size, w) / C(n, w) -- probability that all w
    positions of a uniformly random w-subset land inside a FIXED
    union_size-element subset. math.comb(union_size, w) is 0 (not an error)
    when union_size < w, correctly encoding "impossible" with no special case.

    Raises:
        ValueError: if n<0, w outside [0,n], or union_size outside [0,n].
    """
    _validate_n_w(n, w)
    if not (0 <= union_size <= n):
        raise ValueError(f"union_size must be in [0, {n}], got {union_size}")
    return comb(union_size, w) / comb(n, w)


# =============================================================================
# SECTION 3 -- SELF-CHECKS
# Identities this file depends on. Run BEFORE any output; a violation means
# nothing in Section 4 should be believed.
# =============================================================================


def run_self_checks() -> None:
    """Assert every identity this file depends on.

    Raises:
        AssertionError: on any violated invariant.
    """
    n, w = N_CANONICAL, W_CANONICAL

    # 1. Vandermonde identity: partitioning C(n,w) by overlap-with-reference b
    #    must recover C(n,w) exactly (property 1's acceptance criterion; the
    #    identity this file's ACCEPTANCE break (c) is designed to violate).
    total = comb(n, w)
    assert sum_overlap_set_cardinalities(n, w) == total, (
        f"Vandermonde identity violated at n={n}, w={w}: sum(|Omega(n,w,b)|) != C(n,w)"
    )

    # 2. False-positive probability reproduces the four externally-verified
    #    values, to the stated tolerance (M16: exact value, not a loose bound).
    for theta, expected in _VERIFIED_FALSE_POSITIVE_AT_THETA.items():
        got = false_positive_probability(n, w, theta)
        rel_err = abs(got - expected) / expected
        assert rel_err < _FALSE_POSITIVE_REL_TOL, (
            f"false_positive_probability(n={n},w={w},theta={theta})={got:.6e} disagrees "
            f"with the verified value {expected:.6e} (relative error {rel_err:.2e})"
        )

    # 3. Expected overlap: exact closed form w^2/n.
    assert expected_overlap(n, w) == w * w / n

    # 4. M2 lesson: capacity_patterns is a COUNT (85 digits at n=2048,w=40);
    #    capacity_bits is its log2 (a few hundred bits) -- never confuse the two.
    patterns = capacity_patterns(n, w)
    bits = capacity_bits(n, w)
    assert len(str(patterns)) == 85, f"C({n},{w}) has {len(str(patterns))} digits, expected 85"
    direct_bits = log2(float(patterns))  # safe: float(patterns) ~ 1e84, well under float64 max
    assert abs(bits - direct_bits) < 1e-6, (
        f"lgamma-based capacity_bits={bits} disagrees with direct log2(float(C(n,w)))={direct_bits}"
    )
    assert bits < 300 < patterns, (
        "capacity_bits must be orders of magnitude smaller than capacity_patterns "
        "(M2: a pattern COUNT is not a bit count)"
    )

    # 5. bind() is EXACTLY similarity-preserving for a fixed second argument.
    a1 = encode_token("alpha", n, w, seed=1)
    a2 = encode_token("beta", n, w, seed=1)
    role = encode_token("role", n, w, seed=1)
    assert overlap(bind(a1, role), bind(a2, role)) == overlap(a1, a2), (
        "bind() did not preserve similarity: overlap(bind(a1,b),bind(a2,b)) != overlap(a1,a2)"
    )

    # 6. noisy_copy()'s overlap-with-original is exactly w-k, for every k.
    reference = encode_token("noise-reference", n, w, seed=2)
    for k in (0, 1, w // 2, w):
        corrupted = noisy_copy(reference, k, seed=3)
        assert overlap(reference, corrupted) == w - k, (
            f"noisy_copy(k={k}) produced overlap {overlap(reference, corrupted)}, "
            f"expected exactly {w - k}"
        )

    # 7. union()/bundle() edge cases: empty union, single-pattern bundle is a no-op.
    assert union([]) == frozenset()
    solo = encode_token("solo", n, w, seed=4)
    assert bundle([solo], w) == solo, "bundling one pattern at its own w must be a no-op"


# =============================================================================
# SECTION 4 -- REPORT
# Formatting only. No arithmetic beyond calling Section 2.
# =============================================================================


def _heading(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def report_overlap_distribution() -> None:
    _heading("1. OVERLAP SET CARDINALITY -- Vandermonde self-check")
    n, w = N_CANONICAL, W_CANONICAL
    print(f"n={n}, w={w} (Numenta canonical SDR)\n")
    total = sum_overlap_set_cardinalities(n, w)
    print(f"sum_b |Omega(n,w,b)| = {total}")
    print(f"C(n,w)               = {comb(n, w)}")
    print(f"exact match           : {total == comb(n, w)}")


def report_expected_overlap() -> None:
    _heading("2. EXPECTED OVERLAP OF TWO RANDOM SDRs")
    n, w = N_CANONICAL, W_CANONICAL
    print(f"analytic w^2/n = {expected_overlap(n, w):.5f}")


def report_false_positive_table() -> None:
    _heading("3. FALSE POSITIVE PROBABILITY AT MATCH THRESHOLD theta")
    n, w = N_CANONICAL, W_CANONICAL
    print(f"{'theta':>6}{'fp(n,w,theta)':>18}")
    for theta in sorted(_VERIFIED_FALSE_POSITIVE_AT_THETA, reverse=True):
        print(f"{theta:>6}{false_positive_probability(n, w, theta):>18.4e}")


def report_capacity() -> None:
    _heading("4. CAPACITY -- COUNT vs BITS (mistake M2)")
    n, w = N_CANONICAL, W_CANONICAL
    patterns = capacity_patterns(n, w)
    bits = capacity_bits(n, w)
    digits = len(str(patterns))
    print(f"capacity_patterns(n,w) = C({n},{w}) has {digits} digits (~10^{digits - 1})")
    print(f"capacity_bits(n,w)     = log2(C({n},{w})) = {bits:.2f} bits")
    print("These are DIFFERENT QUANTITIES: the first is an address-space size, the")
    print("second is what the substrate can actually carry (09_mistake_log.md, M2).")


def report_worked_example() -> None:
    _heading("5. ONE CONCRETE INPUT, TO EYEBALL")
    token = "Kelmoran Vushiel-4417"
    sdr = encode_token(token, N_CANONICAL, W_CANONICAL, seed=42)
    print(f"encode_token({token!r}, n={N_CANONICAL}, w={W_CANONICAL}, seed=42)")
    print(f"  len(active)             = {len(sdr.active)}")
    print(f"  first 10 active indices = {sdr.active[:10]}")


def main() -> int:
    run_self_checks()
    report_overlap_distribution()
    report_expected_overlap()
    report_false_positive_table()
    report_capacity()
    report_worked_example()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
