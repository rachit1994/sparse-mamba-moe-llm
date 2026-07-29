"""Dataset B: structured synthetic biographies with exact, discoverable latent structure.

THE QUESTION this file answers: can a dataset have real, exploitable regularity
(unlike Dataset A's independent-uniform attributes, which have none by
construction) while its entropy STAYS exactly computable in closed form?
See implementation/08_thesis_alignment.md sections 2-3 for why this dataset is
thesis-critical: on Dataset A, Shannon guarantees a pattern-forming substrate
and a lookup table tie, because there is nothing to discover.

CONSTRUCTION
    K latent archetypes. Archetype k fixes, per attribute j, a size-D "support"
    of candidate values (D = 2**deviation_bits). Person generation: draw
    k ~ Uniform{0..K-1}; independently for every attribute j, draw the
    person's value uniformly from archetype k's support for j. This creates
    real cross-attribute correlation (two people sharing an archetype tend to
    share attribute values), which is exactly what Dataset A lacks.

CHOSEN: option (a) from the brief -- DISJOINT SUPPORTS, not (b) collision
accounting. For every attribute, the K archetypes' D-sized supports are
pairwise-disjoint subsets of that attribute's value vocabulary (built by
`_archetype_supports` as non-overlapping slices of a permutation of the
attribute's domain). This requires `n_archetypes * 2**deviation_bits <=`
every attribute's cardinality; the tightest constraint is "major" at V=100
(`schema.ATTRIBUTE_CARDINALITY`). `_validate_structured_config` checks this
eagerly and names the offending attribute if it is not satisfiable.

WHY DISJOINT SUPPORTS MAKE THE ENTROPY FORMULA EXACT -- the proof the brief
asks for:
    Let A be the person's archetype (LATENT -- never written to `Person`) and
    V = (V_1, ..., V_J) the J observed attribute values. By construction:
        H(A)          = log2(K)         -- uniform archetype draw
        H(V_j | A)    = log2(D)         -- uniform draw within a D-sized
                                            support, same size for every k
        V_1..V_J are drawn independently GIVEN A (one RNG stream per
        attribute; see `_iter_structured_people`)
    so by the chain rule:
        H(A, V) = H(A) + sum_j H(V_j | A) = log2(K) + J * deviation_bits.
    The dataset records only V, not A, so the quantity that matters is H(V):
        H(V) = H(A, V) - H(A | V).
    Disjoint supports mean two different archetypes can NEVER produce the
    same value for a given attribute, so A is a DETERMINISTIC function of any
    single observed V_j (look up which archetype's block contains it) --
    therefore H(A | V) = 0 exactly, and
        H(V) = H(A, V) = log2(K) + J * deviation_bits,
    with no approximation. This is `_bits_per_structured_person`'s formula.

    If supports could overlap, H(A | V) > 0 strictly and H(V) would be
    strictly LESS than log2(K) + J*deviation_bits -- the naive formula would
    OVERSTATE entropy, silently inflating every downstream bits/substrate
    number (the exact failure mode 08_thesis_alignment.md section 2 and this
    module's brief both warn about). `_archetype_supports` asserts
    disjointness explicitly rather than only relying on it following from
    slicing a permutation, so a future refactor that breaks the invariant
    fails loudly instead of silently overstating H(V).

HOW TO REVIEW: check `_archetype_supports`'s disjointness assertion and the
proof above against each other; then check `_bits_per_structured_person`
implements exactly the formula derived above. Everything else is streaming
plumbing reused from `src.data.generate` (identity assignment, attribute
value rendering) so that a Dataset B `Person` is byte-for-byte the same
shape Dataset A produces -- `generate.render_training_text` and
`generate.render_probe_qa` work on it unmodified, and it is not reimplemented
here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Final

import numpy as np

from src.data import generate as g
from src.data.schema import ATTRIBUTE_CARDINALITY, ATTRIBUTES, BITS_PER_PERSON, Person

__all__ = [
    "generate_structured_people",
    "structured_entropy_bits",
    "compression_ratio",
]

# Fixed structural constant: WHICH domain indices each archetype owns for a
# given (n_archetypes, deviation_bits) config is part of the dataset's fixed
# "world", not controlled by the caller's --seed -- mirrors generate.py's own
# _VOCAB_SEED convention ("the seed picks who, not what the universe of
# possible values is"). Deliberately distinct from generate._VOCAB_SEED so
# the two seed spaces cannot be confused for the same purpose.
_ARCHETYPE_VOCAB_SEED: Final[int] = 0xDECADE0

_GEN_CHUNK: Final[int] = 4096  # matches generate.py's vectorised-batch size


def _validate_structured_config(n_archetypes: int, deviation_bits: int) -> None:
    """Shared eager validation for (n_archetypes, deviation_bits).

    Raises:
        TypeError: n_archetypes or deviation_bits is not an int (e.g. bool or
            float) -- this project requires typed errors for malformed
            inputs, not silent coercion.
        ValueError: n_archetypes < 1, deviation_bits < 0, or
            n_archetypes * 2**deviation_bits exceeds some attribute's
            cardinality -- there would not be enough distinct values to build
            the disjoint per-archetype supports the entropy proof above
            depends on.
    """
    if isinstance(n_archetypes, bool) or not isinstance(n_archetypes, int):
        raise TypeError(f"n_archetypes must be an int, got {type(n_archetypes).__name__}")
    if isinstance(deviation_bits, bool) or not isinstance(deviation_bits, int):
        raise TypeError(f"deviation_bits must be an int, got {type(deviation_bits).__name__}")
    if n_archetypes < 1:
        raise ValueError(f"n_archetypes must be >= 1, got {n_archetypes}")
    if deviation_bits < 0:
        raise ValueError(f"deviation_bits must be >= 0, got {deviation_bits}")
    needed = n_archetypes * (2**deviation_bits)
    for attribute, cardinality in ATTRIBUTE_CARDINALITY.items():
        if needed > cardinality:
            raise ValueError(
                f"n_archetypes={n_archetypes} * 2**deviation_bits={2**deviation_bits} = "
                f"{needed} disjoint values are required per attribute, but "
                f"{attribute!r} has only {cardinality} possible values -- reduce "
                f"n_archetypes or deviation_bits"
            )


def _bits_per_structured_person(n_archetypes: int, deviation_bits: int) -> float:
    """Exact per-person entropy of Dataset B, in bits: log2(K) + J*deviation_bits.

    Exact -- not approximate -- because archetype value supports are
    pairwise disjoint per attribute; see the module docstring's proof.
    """
    return math.log2(n_archetypes) + len(ATTRIBUTES) * deviation_bits


def _archetype_supports(attribute: str, n_archetypes: int, deviation_bits: int) -> np.ndarray:
    """Pairwise-disjoint per-archetype domain-index blocks for `attribute`.

    Returns an (n_archetypes, 2**deviation_bits) int64 array: row k holds
    archetype k's domain indices into `attribute`'s value vocabulary. Column 0
    is archetype k's "preferred" value; the remaining columns are deviation
    alternatives -- a person of archetype k lands on column 0 with probability
    1/2**deviation_bits and "deviates" otherwise. Rows are pairwise disjoint
    by construction: they are non-overlapping slices of a permutation of
    range(cardinality), and a permutation has no repeated elements, so no two
    slices can share an index. This disjointness is what makes
    `_bits_per_structured_person` exact -- see the module docstring.

    Raises:
        ValueError: n_archetypes * 2**deviation_bits exceeds `attribute`'s
            cardinality (not enough distinct values to build one disjoint
            block per archetype).
    """
    cardinality = ATTRIBUTE_CARDINALITY[attribute]
    d = 2**deviation_bits
    needed = n_archetypes * d
    if needed > cardinality:
        raise ValueError(
            f"attribute {attribute!r} has cardinality {cardinality}, but "
            f"n_archetypes={n_archetypes} * 2**deviation_bits={d} = {needed} "
            f"disjoint values are required to build one support block per "
            f"archetype; reduce n_archetypes or deviation_bits"
        )
    perm_seed = g._derive_seed(  # noqa: SLF001 -- deliberate reuse, see module docstring
        _ARCHETYPE_VOCAB_SEED, f"{attribute}:{n_archetypes}:{deviation_bits}"
    )
    rng = np.random.default_rng(perm_seed)
    blocks = rng.permutation(cardinality)[:needed].reshape(n_archetypes, d)

    # Defence in depth (mistake-log M5 territory: an overstated entropy is
    # silent, with no exception and no red test). A permutation is disjoint
    # by definition, but assert it anyway so a future refactor of this
    # slicing logic cannot silently break the one invariant the entropy
    # formula depends on.
    if np.unique(blocks).size != blocks.size:
        raise AssertionError(
            f"archetype supports for {attribute!r} are not pairwise disjoint; "
            f"structured_entropy_bits would overstate the true entropy (see "
            f"module docstring proof)"
        )
    return blocks


def _value_lookup(attribute: str) -> Callable[[int], str]:
    """Return a domain-index -> value-string function for `attribute`.

    `birth_date` is closed-form in generate.py (day-of-year x year; its
    73,000-entry vocabulary is never materialised as a tuple). Every other
    attribute has a materialised vocabulary tuple. Both branches call into
    generate.py's own rendering rather than reimplementing it, so Dataset B's
    values are byte-identical in format to Dataset A's.
    """
    if attribute == "birth_date":

        def lookup(domain_index: int) -> str:
            day = domain_index % g._DAYS_PER_YEAR + 1  # noqa: SLF001
            year = domain_index // g._DAYS_PER_YEAR + g._YEAR_MIN  # noqa: SLF001
            return g._format_birth_date(day, year)  # noqa: SLF001

        return lookup
    vocab = g._attribute_value_vocab(attribute)  # noqa: SLF001
    return lambda domain_index: vocab[domain_index]


def _iter_structured_people(
    n_people: int,
    name_key: int,
    n_archetypes: int,
    deviation_bits: int,
    supports: dict[str, np.ndarray],
    lookups: dict[str, Callable[[int], str]],
    archetype_rng: np.random.Generator,
    deviation_rngs: dict[str, np.random.Generator],
) -> Iterator[Person]:
    """Core streaming loop: person index i gets archetype k, then an
    independent-given-k deviation draw per attribute. O(1) memory per
    yielded person (batches RNG calls in chunks of `_GEN_CHUNK`, mirroring
    generate.py's `_iter_people`), so this scales the same way Dataset A does.
    """
    d = 2**deviation_bits
    index = 0
    remaining = n_people
    while remaining > 0:
        chunk = min(_GEN_CHUNK, remaining)
        archetype_draws = archetype_rng.integers(0, n_archetypes, size=chunk)
        deviation_draws = {
            attr: deviation_rngs[attr].integers(0, d, size=chunk) for attr in ATTRIBUTES
        }
        for i in range(chunk):
            k = int(archetype_draws[i])
            attributes = {
                attr: lookups[attr](int(supports[attr][k, deviation_draws[attr][i]]))
                for attr in ATTRIBUTES
            }
            yield Person(person_id=g._make_person_id(index, name_key), attributes=attributes)  # noqa: SLF001
            index += 1
        remaining -= chunk


def generate_structured_people(
    n_people: int, seed: int, n_archetypes: int, deviation_bits: int
) -> Iterator[Person]:
    """Deterministically stream `n_people` Dataset B (structured) individuals.

    Same `Person` type, same attribute vocabularies, and the same identity
    (name) assignment machinery as `generate.generate_people` -- only how
    attribute VALUES are drawn differs (shared-archetype + deviation instead
    of independent uniform). See the module docstring for the construction
    and its entropy proof.

    Args:
        n_people: number of people to generate. Non-negative int.
        seed: determinism seed. Controls archetype assignment, per-attribute
            deviation draws, and (via generate.py's own machinery) person
            identities -- NOT archetype value supports, which are a fixed
            structural property of (n_archetypes, deviation_bits); see
            `_archetype_supports`.
        n_archetypes: K, number of latent archetypes. Int >= 1.
        deviation_bits: bits of within-archetype deviation entropy PER
            ATTRIBUTE, applied uniformly to every attribute. Int >= 0.

    Returns:
        An iterator yielding Person objects for index 0..n_people-1, in order.

    Raises:
        TypeError: n_people, seed, n_archetypes, or deviation_bits is not an
            int.
        ValueError: n_people < 0, seed < 0, n_people exceeds the shared
            name-space capacity (see `generate.generate_people`),
            n_archetypes < 1, deviation_bits < 0, or the config cannot be
            realised with disjoint per-archetype supports (see
            `_validate_structured_config`).
    """
    g._validate_n_people_and_seed(n_people, seed)  # noqa: SLF001
    _validate_structured_config(n_archetypes, deviation_bits)
    if n_people > g._NAME_DOMAIN_SIZE:  # noqa: SLF001
        raise ValueError(
            f"n_people={n_people} exceeds the synthetic name-space capacity "
            f"({g._NAME_DOMAIN_SIZE}); cannot guarantee unique identities"  # noqa: SLF001
        )

    supports = {
        attr: _archetype_supports(attr, n_archetypes, deviation_bits) for attr in ATTRIBUTES
    }
    lookups = {attr: _value_lookup(attr) for attr in ATTRIBUTES}
    name_key, _ = g._derive_name_key_and_attr_seeds(seed)  # noqa: SLF001
    archetype_rng = np.random.default_rng(g._derive_seed(seed, "structured_archetype"))  # noqa: SLF001
    deviation_rngs = {
        attr: np.random.default_rng(g._derive_seed(seed, f"structured_deviation:{attr}"))  # noqa: SLF001
        for attr in ATTRIBUTES
    }
    return _iter_structured_people(
        n_people,
        name_key,
        n_archetypes,
        deviation_bits,
        supports,
        lookups,
        archetype_rng,
        deviation_rngs,
    )


def structured_entropy_bits(n_people: int, n_archetypes: int, deviation_bits: int) -> float:
    """Total entropy of `n_people` Dataset B people, in bits. Exact -- see the
    module docstring proof; mirrors `schema.dataset_entropy_bits`'s contract.

    Raises:
        TypeError: n_archetypes or deviation_bits is not an int.
        ValueError: n_people < 0, n_archetypes < 1, deviation_bits < 0, or the
            config cannot be realised with disjoint per-archetype supports.
    """
    _validate_structured_config(n_archetypes, deviation_bits)
    if n_people < 0:
        raise ValueError(f"n_people must be non-negative, got {n_people}")
    return n_people * _bits_per_structured_person(n_archetypes, deviation_bits)


def compression_ratio(n_archetypes: int, deviation_bits: int) -> float:
    """How many times smaller Dataset B's per-person entropy is than Dataset
    A's: `schema.BITS_PER_PERSON / _bits_per_structured_person(...)`.

    ratio > 1 exactly when the archetype+deviation encoding of a person is
    smaller than encoding every attribute independently -- the compressible
    structure the thesis needs Dataset B to have
    (implementation/08_thesis_alignment.md section 3). Falls as either
    n_archetypes or deviation_bits rises (more bits needed to describe a
    person shrinks the gap to Dataset A); the ratio is undefined (raises)
    only in the degenerate n_archetypes=1, deviation_bits=0 case, where
    Dataset B carries zero bits per person.

    Raises:
        TypeError: n_archetypes or deviation_bits is not an int.
        ValueError: n_archetypes < 1, deviation_bits < 0, the config cannot be
            realised with disjoint per-archetype supports, or Dataset B's
            per-person entropy is exactly 0 (division by zero).
    """
    _validate_structured_config(n_archetypes, deviation_bits)
    bits_per_person_b = _bits_per_structured_person(n_archetypes, deviation_bits)
    if bits_per_person_b == 0:
        raise ValueError(
            "n_archetypes=1 and deviation_bits=0 gives Dataset B zero bits per "
            "person; compression ratio against Dataset A is undefined"
        )
    return BITS_PER_PERSON / bits_per_person_b
