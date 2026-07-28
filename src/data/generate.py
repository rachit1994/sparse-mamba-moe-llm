"""Synthetic biography dataset generator for the knowledge-capacity experiment.

Generates `N` synthetic individuals with independently, near-uniformly sampled
attributes at the cardinalities fixed by `src/data/schema.py`, and streams four
JSONL splits (train / probe / unseen_person / sealed) to disk without ever
materialising the whole dataset in memory.

Design summary (see implementation/04_golden_dataset.md, the spec this file
implements, for the "why"):

* **Identity assignment is a bijection, not a random draw.** Person index `i`
  is mapped to a name via a keyed Feistel permutation (`_feistel_permute`)
  over the full name-space domain. A bijection restricted to any subset of
  distinct inputs is injective, so two different indices can *never* produce
  the same name -- this is a mathematical guarantee, not a birthday-bound
  probabilistic one (implementation/03_edge_cases_and_scares.md: "Name
  collisions are the sneaky one... Assert uniqueness"). Non-cryptographic by
  design (this is a dataset-construction tool, not a security primitive).
* **Attribute value vocabularies are fixed constants**, generated once at
  import time from a hardcoded internal seed (`_VOCAB_SEED`) -- NOT from the
  caller's `--seed`. This mirrors `schema.ATTRIBUTE_CARDINALITY` being a fixed
  constant: the `--seed` argument controls *which* person gets *which* value
  and *which* people land in which split, not what the universe of possible
  values is. This keeps different seeds' datasets drawn from the same
  "world" and keeps vocab-cardinality bugs a one-time, seed-independent
  concern.
* **Attribute values are independent draws** from dedicated `numpy` random
  streams (`numpy.random.SeedSequence.spawn`), one stream per attribute, so
  no attribute can be statistically entangled with another by construction.
* **All non-name attribute values are single-fused invented words** (root +
  suffix syllable, no shared literal words with template text) so that no
  training/probe template can accidentally contain a value-identifying token
  (implementation/04_golden_dataset.md failure-mode table: "Paraphrase
  templates leak the answer").
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np

from src.data.schema import (
    ATTRIBUTE_CARDINALITY,
    ATTRIBUTES,
    BITS_PER_PERSON,
    Person,
    dataset_entropy_bits,
)

__all__ = [
    "generate_people",
    "render_training_text",
    "render_probe_qa",
    "write_dataset",
]

# ---------------------------------------------------------------------------
# Synthetic word banks. Every bank below is an invented syllable/word list --
# none of it is drawn from a real-world gazetteer, university directory, or
# name corpus. This is the contamination proof for R1 (04_golden_dataset.md
# section 1): "no real person representable", by construction.
# ---------------------------------------------------------------------------

_VOCAB_SEED: Final[int] = 0xC0FFEE

# -- person identity: "<FirstName> <LastName>-<4 digits>" ------------------
_NAME_FIRST_A: Final[tuple[str, ...]] = (
    "Kel", "Mor", "Van", "Ash", "Bri", "Tor", "Sel", "Dun", "Fen", "Gor",
    "Hal", "Iven", "Jor", "Kav", "Lom", "Myr", "Nor", "Oren", "Pel", "Ren",
    "Sav", "Tev", "Ulv", "Wyn",
)
_NAME_FIRST_B: Final[tuple[str, ...]] = (
    "an", "oran", "ric", "dor", "wyn", "iel", "ael", "oth", "und", "ard",
    "eth", "ion", "van", "wick", "mund", "gard", "lyn", "sten", "rick", "vane",
    "dren", "mara", "cott", "field",
)
_NAME_LAST_A: Final[tuple[str, ...]] = (
    "Vush", "Bral", "Corw", "Dresh", "Eldar", "Fenwick", "Grael", "Holt",
    "Ith", "Jarek", "Kestrel", "Lysander", "Maren", "Nyx", "Orin", "Pryce",
    "Quill", "Roth", "Sarn", "Thale", "Ulric", "Varek", "Wraith", "Yorick",
)
_NAME_LAST_B: Final[tuple[str, ...]] = (
    "iel", "ovic", "ander", "ley", "mont", "ford", "stead", "holm", "wick",
    "garde", "mere", "vale", "thorn", "crest", "haven", "ridge", "brook",
    "fell", "moor", "glen", "dale", "spire", "reach", "hollow",
)
_NAME_SUFFIX_DIGITS: Final[int] = 10_000  # the "-4417" part

_FIRST_COMBOS: Final[int] = len(_NAME_FIRST_A) * len(_NAME_FIRST_B)
_LAST_COMBOS: Final[int] = len(_NAME_LAST_A) * len(_NAME_LAST_B)
_NAME_DOMAIN_SIZE: Final[int] = _FIRST_COMBOS * _LAST_COMBOS * _NAME_SUFFIX_DIGITS

# -- attribute value banks (root, suffix) -----------------------------------
_CITY_ROOT: Final[tuple[str, ...]] = (
    "Tarn", "Bren", "Ashk", "Wyv", "Corr", "Dun", "Elm", "Fenr", "Grim", "Holt",
    "Ivor", "Jask", "Korr", "Lorn", "Myr", "Norr", "Osk", "Pell", "Quor", "Rask",
    "Sorn", "Torv", "Ulm", "Vask", "Wren", "Xen", "Yorn", "Zeph", "Brack", "Crell",
    "Dresk", "Ehr", "Frost", "Gorr", "Hesk", "Ithe", "Jorv", "Krell", "Lysk", "Morv",
)
_CITY_SUFFIX: Final[tuple[str, ...]] = (
    "vale", "ford", "haven", "stead", "burg", "holt", "field", "gate", "mere",
    "ridge", "moor", "dale", "crest", "hollow", "port", "ton", "wick", "bridge",
    "spire", "reach", "glen", "shire", "cove", "bury", "forge", "brook", "hurst",
    "land", "well", "ness",
)

_UNI_ROOT: Final[tuple[str, ...]] = (
    "Brindle", "Ashcombe", "Wyndham", "Corvale", "Duskmere", "Elmsworth",
    "Fenwick", "Graystone", "Holloway", "Ironvale", "Jasperfield", "Kestrelmoor",
    "Larkspur", "Millbrook", "Norwyck", "Oakhaven", "Pemberton", "Quillfeld",
    "Ravensburg", "Silverdell", "Thornfield", "Underhill", "Vesperton",
    "Wickerbrook", "Yewcross", "Brackenmoor", "Crestvale", "Drossmere",
    "Everhold", "Fallowridge",
)
_UNI_SUFFIX: Final[tuple[str, ...]] = (
    "court", "holt", "mere", "forge", "spire", "hall", "cross", "field",
    "gate", "haven", "ridge", "view", "house", "grove", "bourne",
)

_MAJOR_ROOT: Final[tuple[str, ...]] = (
    "Astro", "Xeno", "Hydro", "Cryo", "Petro", "Bio", "Photo", "Litho",
    "Aero", "Thermo", "Geo", "Micro", "Macro", "Chrono", "Necro", "Pyro",
    "Helio", "Morpho", "Neuro", "Techno",
)
_MAJOR_SUFFIX: Final[tuple[str, ...]] = (
    "logy", "nomics", "graphy", "technics", "urgy", "chemistry", "metrics",
    "dynamics", "genetics", "physics", "architecture", "synthesis",
)

_EMPLOYER_ROOT: Final[tuple[str, ...]] = (
    "Halcyon", "Verun", "Kestrion", "Thornwell", "Ashgate", "Brindlewood",
    "Corvid", "Duskwatch", "Emberlyn", "Frostholt", "Gravemark", "Hollowpine",
    "Ironbrand", "Jasperwind", "Korrigan", "Larkfield", "Mistvale", "Norrath",
    "Oakspire", "Pellamor", "Quorvane", "Ravensmere", "Silverfen", "Torvane",
    "Umbral", "Vespera", "Wyrmhold", "Xanthe", "Yewbrook", "Zenithal",
    "Brackwater", "Crellmont", "Driftshade", "Everspring", "Fallowmere",
    "Grimshard", "Hallowick", "Ithermere", "Jorvane", "Krellstone",
)
_EMPLOYER_SUFFIX: Final[tuple[str, ...]] = (
    "Vornex", "Thalyx", "Mekar", "Droscen", "Kaivex", "Zinthar", "Vexil",
    "Hollomyn", "Draltek", "Quenvar", "Fynor", "Fyrion", "Gravash", "Nexil",
    "Orinth", "Vashkar", "Kestryn", "Dornel", "Zaelith", "Phenrix", "Crinval",
    "Tholmex", "Vyrexis", "Sarnith", "Glimmox", "Nethyra", "Corinox", "Axelyn",
    "Varnik", "Ostryl",
)

# -- birth_date: day-of-synthetic-year (1..365, no leap day) x year --------
# 365 * 200 == ATTRIBUTE_CARDINALITY["birth_date"] == 73_000, exactly.
_MONTH_NAMES: Final[tuple[str, ...]] = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
_MONTH_DAYS: Final[tuple[int, ...]] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_DAYS_PER_YEAR: Final[int] = sum(_MONTH_DAYS)  # 365
_YEAR_MIN: Final[int] = 1900
_YEAR_MAX: Final[int] = 2099  # inclusive
_YEARS_SPAN: Final[int] = _YEAR_MAX - _YEAR_MIN + 1  # 200

if _DAYS_PER_YEAR * _YEARS_SPAN != ATTRIBUTE_CARDINALITY["birth_date"]:
    raise AssertionError(
        f"birth_date encoding ({_DAYS_PER_YEAR} days x {_YEARS_SPAN} years = "
        f"{_DAYS_PER_YEAR * _YEARS_SPAN}) no longer matches "
        f"schema.ATTRIBUTE_CARDINALITY['birth_date']={ATTRIBUTE_CARDINALITY['birth_date']}; "
        f"the entropy formula would silently be wrong."
    )


def _format_birth_date(day_of_year: int, year: int) -> str:
    """Render a (day_of_year, year) pair as "<day> <Month> <year>".

    Args:
        day_of_year: 1..365 inclusive.
        year: 1900..2099 inclusive.

    Returns:
        E.g. "12 March 1987".

    Raises:
        ValueError: if day_of_year or year is out of range.
    """
    if not (1 <= day_of_year <= _DAYS_PER_YEAR):
        raise ValueError(f"day_of_year must be in [1, {_DAYS_PER_YEAR}], got {day_of_year}")
    if not (_YEAR_MIN <= year <= _YEAR_MAX):
        raise ValueError(f"year must be in [{_YEAR_MIN}, {_YEAR_MAX}], got {year}")
    remaining = day_of_year
    for month_idx, days_in_month in enumerate(_MONTH_DAYS):
        if remaining <= days_in_month:
            return f"{remaining} {_MONTH_NAMES[month_idx]} {year}"
        remaining -= days_in_month
    raise AssertionError("unreachable: day_of_year loop exhausted _MONTH_DAYS")  # pragma: no cover


def _build_fused_vocab(
    bank_a: tuple[str, ...], bank_b: tuple[str, ...], count: int, seed: int, *, sep: str = ""
) -> tuple[str, ...]:
    """Deterministically build `count` unique `bank_a[i]{sep}bank_b[j]` strings.

    Uniqueness is exact (not probabilistic): `count` distinct (i, j) index
    pairs are drawn via a full permutation of the len(bank_a)*len(bank_b)
    index domain, so no two chosen pairs can coincide. The resulting fused
    strings are then verified unique defensively (a hand-picked syllable bank
    could in principle produce two different (i, j) pairs that fuse to the
    same string, e.g. "Kel"+"van" == "Kelv"+"an" for a poorly chosen bank;
    this is a real risk for concatenation-based vocab construction, so it is
    checked, not assumed).

    Args:
        bank_a: first-component word bank.
        bank_b: second-component word bank.
        count: how many unique values to produce. Must be <= len(bank_a) * len(bank_b).
        seed: determinism seed for the internal permutation (NOT the caller's
            --seed -- see module docstring: vocabularies are seed-independent
            constants of this generator).
        sep: separator between the two components (e.g. " " for two-word
            values, "" for a single fused word).

    Returns:
        `count` unique strings, in an arbitrary but deterministic order.

    Raises:
        ValueError: if count exceeds the bank's combinatorial domain.
        RuntimeError: if the fused strings are not all unique (a hand-authored
            word-bank bug -- ambiguous concatenation -- not a runtime data bug).
    """
    domain = len(bank_a) * len(bank_b)
    if count > domain:
        raise ValueError(
            f"requested {count} unique values but bank_a x bank_b domain is only {domain}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(domain)[:count]
    values = tuple(f"{bank_a[int(idx) // len(bank_b)]}{sep}{bank_b[int(idx) % len(bank_b)]}" for idx in chosen)
    if len(set(values)) != count:
        raise RuntimeError(
            "vocabulary word banks produced ambiguous (colliding) fused strings; "
            "the syllable banks must be revised so no two (root, suffix) pairs "
            "concatenate to the same string"
        )
    return values


_CITY_VALUES: Final[tuple[str, ...]] = _build_fused_vocab(
    _CITY_ROOT, _CITY_SUFFIX, ATTRIBUTE_CARDINALITY["birth_city"], seed=_VOCAB_SEED + 1
)
_UNIVERSITY_VALUES: Final[tuple[str, ...]] = _build_fused_vocab(
    _UNI_ROOT, _UNI_SUFFIX, ATTRIBUTE_CARDINALITY["university"], seed=_VOCAB_SEED + 2
)
_MAJOR_VALUES: Final[tuple[str, ...]] = _build_fused_vocab(
    _MAJOR_ROOT, _MAJOR_SUFFIX, ATTRIBUTE_CARDINALITY["major"], seed=_VOCAB_SEED + 3
)
_EMPLOYER_VALUES: Final[tuple[str, ...]] = _build_fused_vocab(
    _EMPLOYER_ROOT, _EMPLOYER_SUFFIX, ATTRIBUTE_CARDINALITY["employer"], seed=_VOCAB_SEED + 4, sep=" "
)

for _attr_name, _values in (
    ("birth_city", _CITY_VALUES),
    ("university", _UNIVERSITY_VALUES),
    ("major", _MAJOR_VALUES),
    ("employer", _EMPLOYER_VALUES),
):
    if len(_values) != ATTRIBUTE_CARDINALITY[_attr_name]:
        raise AssertionError(
            f"{_attr_name} vocabulary has {len(_values)} values, expected "
            f"schema.ATTRIBUTE_CARDINALITY[{_attr_name!r}]={ATTRIBUTE_CARDINALITY[_attr_name]}"
        )
del _attr_name, _values


def _attribute_value_vocab(attribute: str) -> tuple[str, ...]:
    """Return the fixed, exact vocabulary of possible values for `attribute`.

    Not defined for "birth_date": its 73,000-value vocabulary is a closed-form
    product (365 days x 200 years) and is never materialised as a tuple.

    Raises:
        ValueError: if attribute is not one of the four non-date attributes.
    """
    vocab = {
        "birth_city": _CITY_VALUES,
        "university": _UNIVERSITY_VALUES,
        "major": _MAJOR_VALUES,
        "employer": _EMPLOYER_VALUES,
    }.get(attribute)
    if vocab is None:
        raise ValueError(
            f"no materialised vocabulary for {attribute!r} (birth_date is closed-form; "
            f"expected one of 'birth_city', 'university', 'major', 'employer')"
        )
    return vocab


# ---------------------------------------------------------------------------
# Templates. TRAIN templates are declarative; PROBE templates are held-out
# interrogative phrasings, structurally disjoint (verified by tests/test_data_generate.py
# test 5) from the TRAIN pool. Every template contains only "{name}" and,
# for TRAIN only, "{value}" placeholders -- no static word here is drawn from
# the invented syllable banks above, so no template can leak a value token
# (verified by test 6).
# ---------------------------------------------------------------------------

_TRAIN_TEMPLATES: Final[dict[str, tuple[str, ...]]] = {
    "birth_date": (
        "{name} was born on {value}.",
        "{name} came into the world on {value}.",
        "{name}'s date of birth is {value}.",
        "{name} was born {value}.",
        "The birth date of {name} is {value}.",
        "{name} first drew breath on {value}.",
    ),
    "birth_city": (
        "{name} grew up in {value}.",
        "{name} was raised in {value}.",
        "{name} spent their childhood in {value}.",
        "{name} comes from {value}.",
        "{name}'s hometown is {value}.",
        "{name} hails from {value}.",
    ),
    "university": (
        "{name} studied at {value}.",
        "{name} attended {value}.",
        "{name} earned a degree from {value}.",
        "{name} was educated at {value}.",
        "{name} is an alumnus of {value}.",
        "{name} completed their studies at {value}.",
    ),
    "major": (
        "{name} majored in {value}.",
        "{name} studied {value}.",
        "{name}'s field of study was {value}.",
        "{name} specialised in {value}.",
        "{name} focused their studies on {value}.",
        "{name} trained in {value}.",
    ),
    "employer": (
        "{name} works at {value}.",
        "{name} is employed by {value}.",
        "{name} has a job at {value}.",
        "{name} currently works for {value}.",
        "{name} is on the staff of {value}.",
        "{name} holds a position at {value}.",
    ),
}

_PROBE_TEMPLATES: Final[dict[str, tuple[str, ...]]] = {
    "birth_date": (
        "When was {name} born?",
        "What is {name}'s date of birth?",
        "On what date did {name} come into the world?",
        "What is {name}'s birth date?",
        "Can you tell me when {name} was born?",
        "What day was {name} born on?",
    ),
    "birth_city": (
        "Where was {name} born?",
        "What city did {name} grow up in?",
        "{name}'s hometown is which city?",
        "In which city did {name} spend their childhood?",
        "Where does {name} come from?",
        "What is {name}'s birthplace?",
    ),
    "university": (
        "Where did {name} study?",
        "Which university did {name} attend?",
        "What institution granted {name} a degree?",
        "{name} is an alumnus of which university?",
        "Where was {name} educated?",
        "At which university did {name} complete their studies?",
    ),
    "major": (
        "What did {name} major in?",
        "What was {name}'s field of study?",
        "In what subject did {name} specialise?",
        "What did {name} study at university?",
        "What subject did {name} focus their studies on?",
        "What was {name} trained in?",
    ),
    "employer": (
        "Where does {name} work?",
        "Who employs {name}?",
        "What company does {name} work for?",
        "Where is {name} currently employed?",
        "What is {name}'s employer?",
        "At what organisation does {name} hold a position?",
    ),
}

for _attr in ATTRIBUTES:
    if len(_TRAIN_TEMPLATES[_attr]) < 5:
        raise AssertionError(f"train templates for {_attr!r} has fewer than 5 entries")
    if len(_PROBE_TEMPLATES[_attr]) < 5:
        raise AssertionError(f"probe templates for {_attr!r} has fewer than 5 entries")
    if set(_TRAIN_TEMPLATES[_attr]) & set(_PROBE_TEMPLATES[_attr]):
        raise AssertionError(f"train/probe template pools for {_attr!r} are not disjoint")
del _attr


# ---------------------------------------------------------------------------
# Identity assignment: keyed, cycle-walking Feistel permutation.
# ---------------------------------------------------------------------------

_MASK64: Final[int] = (1 << 64) - 1


def _mix64(x: int) -> int:
    """A deterministic, non-cryptographic 64-bit integer mixer (splitmix64-style)."""
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x ^= x >> 31
    return x


def _feistel_round(x: int, round_no: int, key: int, out_bits: int) -> int:
    mixed = _mix64((x << 20) ^ (round_no << 12) ^ key ^ 0x9E3779B97F4A7C15)
    return mixed & ((1 << out_bits) - 1)


def _feistel_permute(index: int, domain_size: int, key: int, rounds: int = 4) -> int:
    """Bijectively permute `index` within [0, domain_size) under `key`.

    A keyed, cycle-walking Feistel network: pads to an even bit-width so the
    two halves are equal-width, iterates `rounds` Feistel rounds, and re-applies
    the whole permutation ("cycle-walking") whenever the padded output falls
    outside [0, domain_size) -- a standard format-preserving-permutation
    technique for permuting an arbitrary (non-power-of-two) domain. This is
    NOT a cryptographic construction (4 rounds of a non-cryptographic mixer is
    not secure against a deliberate adversary); it exists solely to give a
    collision-free identity assignment for dataset construction, verified for
    bijectivity by test_data_generate.py.

    Args:
        index: value to permute, must be in [0, domain_size).
        domain_size: size of the domain being permuted. Must be positive.
        key: permutation key (distinguishes independent permutations of the
            same domain, e.g. per --seed).
        rounds: number of Feistel rounds. 4 is ample diffusion for this
            non-adversarial use case.

    Returns:
        A value in [0, domain_size), such that the mapping index -> return
        value is a bijection over [0, domain_size).

    Raises:
        ValueError: if domain_size <= 0 or index is out of [0, domain_size).
    """
    if domain_size <= 0:
        raise ValueError(f"domain_size must be positive, got {domain_size}")
    if not (0 <= index < domain_size):
        raise ValueError(f"index {index} out of range [0, {domain_size})")
    if domain_size == 1:
        return 0

    bits = (domain_size - 1).bit_length()
    if bits % 2 == 1:
        bits += 1
    half = bits // 2
    half_mask = (1 << half) - 1

    x = index
    while True:
        left = (x >> half) & half_mask
        right = x & half_mask
        for round_no in range(rounds):
            left, right = right, left ^ _feistel_round(right, round_no, key, half)
        x = (left << half) | right
        if x < domain_size:
            return x


def _make_person_id(index: int, name_key: int) -> str:
    """Map a person index to a unique "<First> <Last>-<digits>" identity.

    Uniqueness for any set of distinct `index` values (under a fixed
    `name_key`) follows from `_feistel_permute` being a bijection over
    [0, _NAME_DOMAIN_SIZE): distinct inputs to a bijection cannot map to the
    same output.

    Raises:
        ValueError: if index is negative or >= _NAME_DOMAIN_SIZE.
    """
    domain_idx = _feistel_permute(index, _NAME_DOMAIN_SIZE, name_key)
    suffix = domain_idx % _NAME_SUFFIX_DIGITS
    rest = domain_idx // _NAME_SUFFIX_DIGITS
    last_combo = rest % _LAST_COMBOS
    first_combo = rest // _LAST_COMBOS
    first_a, first_b = divmod(first_combo, len(_NAME_FIRST_B))
    last_a, last_b = divmod(last_combo, len(_NAME_LAST_B))
    first_name = f"{_NAME_FIRST_A[first_a]}{_NAME_FIRST_B[first_b]}"
    last_name = f"{_NAME_LAST_A[last_a]}{_NAME_LAST_B[last_b]}"
    return f"{first_name} {last_name}-{suffix:04d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GEN_CHUNK: Final[int] = 4096  # batch size for vectorised attribute sampling


def generate_people(n_people: int, seed: int) -> Iterator[Person]:
    """Deterministically stream `n_people` synthetic individuals.

    Attributes are sampled independently (one dedicated numpy random stream
    per attribute, spawned from `seed` via `numpy.random.SeedSequence.spawn`)
    and near-uniformly, at the exact cardinalities in
    `schema.ATTRIBUTE_CARDINALITY`. Person identities are assigned via a
    keyed bijective permutation over the full name space (see
    `_make_person_id`), so no two people generated under the same `seed` can
    collide, for any `n_people` up to `_NAME_DOMAIN_SIZE` (~3.3 billion).

    Memory: O(1) amortised per yielded person; internally batches attribute
    sampling in chunks of `_GEN_CHUNK` so peak memory does not grow with
    `n_people` (required to stream 4M+ people without OOM).

    Args:
        n_people: number of people to generate. Must be a non-negative int.
        seed: determinism seed. Same (n_people, seed) always yields the same
            sequence of Person objects in the same order. Must be a
            non-negative int.

    Yields:
        Person objects, for index 0..n_people-1 in order.

    Raises:
        TypeError: if n_people or seed is not an int (e.g. a float or bool is
            passed) -- this project requires typed errors, not silent
            coercion, for malformed inputs (implementation/03_edge_cases_and_scares.md).
        ValueError: if n_people < 0, if seed < 0, or if n_people exceeds the
            synthetic name-space capacity (`_NAME_DOMAIN_SIZE`), in which case
            uniqueness could not be guaranteed.
    """
    if isinstance(n_people, bool) or not isinstance(n_people, int):
        raise TypeError(f"n_people must be an int, got {type(n_people).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if n_people < 0:
        raise ValueError(f"n_people must be non-negative, got {n_people}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if n_people > _NAME_DOMAIN_SIZE:
        raise ValueError(
            f"n_people={n_people} exceeds the synthetic name-space capacity "
            f"({_NAME_DOMAIN_SIZE}); cannot guarantee unique identities"
        )

    root = np.random.SeedSequence(seed)
    name_key_seq, day_seq, year_seq, city_seq, uni_seq, major_seq, employer_seq = root.spawn(7)
    name_key = int(name_key_seq.generate_state(1, dtype=np.uint64)[0])
    day_rng = np.random.default_rng(day_seq)
    year_rng = np.random.default_rng(year_seq)
    city_rng = np.random.default_rng(city_seq)
    uni_rng = np.random.default_rng(uni_seq)
    major_rng = np.random.default_rng(major_seq)
    employer_rng = np.random.default_rng(employer_seq)

    n_city = ATTRIBUTE_CARDINALITY["birth_city"]
    n_uni = ATTRIBUTE_CARDINALITY["university"]
    n_major = ATTRIBUTE_CARDINALITY["major"]
    n_employer = ATTRIBUTE_CARDINALITY["employer"]

    index = 0
    remaining = n_people
    while remaining > 0:
        chunk = min(_GEN_CHUNK, remaining)
        days = day_rng.integers(1, _DAYS_PER_YEAR + 1, size=chunk)
        years = year_rng.integers(_YEAR_MIN, _YEAR_MAX + 1, size=chunk)
        city_idx = city_rng.integers(0, n_city, size=chunk)
        uni_idx = uni_rng.integers(0, n_uni, size=chunk)
        major_idx = major_rng.integers(0, n_major, size=chunk)
        employer_idx = employer_rng.integers(0, n_employer, size=chunk)

        for k in range(chunk):
            attributes = {
                "birth_date": _format_birth_date(int(days[k]), int(years[k])),
                "birth_city": _CITY_VALUES[int(city_idx[k])],
                "university": _UNIVERSITY_VALUES[int(uni_idx[k])],
                "major": _MAJOR_VALUES[int(major_idx[k])],
                "employer": _EMPLOYER_VALUES[int(employer_idx[k])],
            }
            yield Person(person_id=_make_person_id(index, name_key), attributes=attributes)
            index += 1
        remaining -= chunk


def render_training_text(person: Person, rng: np.random.Generator) -> list[str]:
    """Render this person's declarative training sentences.

    One sentence per attribute, in `schema.ATTRIBUTES` order, each drawn
    uniformly from that attribute's TRAIN template pool (>=5 templates per
    attribute) via `rng`. This is what gives paraphrase diversity across the
    training corpus -- Allen-Zhu found that with a single fixed phrasing,
    knowledge is stored but not extractable (implementation/04_golden_dataset.md
    section 3).

    Args:
        person: the individual to render.
        rng: a `numpy.random.Generator` controlling template choice. Passed in
            by the caller (not created internally) so this function is a
            pure, deterministic function of its inputs with no hidden global
            RNG state -- required for reproducible streaming generation.

    Returns:
        One sentence per attribute, in `schema.ATTRIBUTES` order.
    """
    sentences: list[str] = []
    for attr in ATTRIBUTES:
        templates = _TRAIN_TEMPLATES[attr]
        template = templates[int(rng.integers(0, len(templates)))]
        sentences.append(template.format(name=person.person_id, value=person.attributes[attr]))
    return sentences


def render_probe_qa(person: Person, attribute: str, template_idx: int) -> tuple[str, str]:
    """Render one held-out-phrasing evaluation question/answer pair.

    Uses the PROBE template pool, a set of interrogative phrasings that is
    structurally disjoint from the declarative TRAIN template pool (asserted
    at import time above) -- so a probe question is guaranteed to be a novel
    surface form the model never saw during training, which is what makes it
    a test of extractable knowledge rather than sentence recall.

    Args:
        person: the individual being asked about.
        attribute: one of `schema.ATTRIBUTES`.
        template_idx: index into that attribute's probe template pool.

    Returns:
        `(question, answer)`. `answer` is the verbatim attribute value string
        (the same string used when rendering training text for this person),
        so exact-match / cross-entropy scoring targets a fixed surface form
        regardless of which probe phrasing was used.

    Raises:
        TypeError: if template_idx is not an int.
        ValueError: if `attribute` is not in `schema.ATTRIBUTES`, or
            `template_idx` is outside the range of that attribute's probe
            template pool.
    """
    if attribute not in ATTRIBUTES:
        raise ValueError(f"unknown attribute {attribute!r}; must be one of {ATTRIBUTES}")
    if isinstance(template_idx, bool) or not isinstance(template_idx, int):
        raise TypeError(f"template_idx must be an int, got {type(template_idx).__name__}")
    templates = _PROBE_TEMPLATES[attribute]
    if not (0 <= template_idx < len(templates)):
        raise ValueError(
            f"template_idx={template_idx} out of range for attribute {attribute!r} "
            f"(has {len(templates)} probe templates)"
        )
    question = templates[template_idx].format(name=person.person_id)
    answer = person.attributes[attribute]
    return question, answer


def _derive_seed(seed: int, label: str) -> int:
    """Deterministically derive an independent, non-negative child seed.

    Used so that `write_dataset`'s auxiliary randomness (split assignment,
    probe/unseen/sealed template selection, the sealed population) is fully
    determined by one top-level `seed` while remaining statistically
    independent -- via SHA-256 -- of every other stream derived the same way.
    This is the mechanism behind the "sealed" split's "separate seed"
    requirement (implementation/04_golden_dataset.md section 5): sealed reuses
    the caller's `seed` but through a distinct label, so it is a fully
    determined, reproducible, yet independent population.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes (bounded memory regardless of file size)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_TRAIN_FRACTION: Final[float] = 0.9


def write_dataset(out_dir: Path, n_people: int, seed: int) -> dict:
    """Stream a full dataset (train/probe/unseen_person/sealed) to `out_dir`.

    Writes five files under `out_dir`:
      - `train.jsonl`: one line per train-split person,
        `{"person_id": ..., "text": "<sentence 1> <sentence 2> ..."}` -- a
        declarative biography paragraph, one sentence per attribute.
      - `probe.jsonl`: one line per (train-split person, attribute),
        `{"person_id", "attribute", "template_idx", "question", "answer"}` --
        held-out-phrasing questions about the *same* people seen in training
        (Tier A capacity probes, implementation/04_golden_dataset.md section 6).
      - `unseen_person.jsonl`: same shape as probe.jsonl, but for the 10% of
        people who never appear in train.jsonl at all -- the leakage control
        that must score ~0.
      - `sealed.jsonl`: same shape as probe.jsonl, but for an entirely
        separate population of `n_people` people generated from a seed
        independently derived from `seed` (see `_derive_seed`), used only at
        phase gates.
      - `manifest.json`: counts, entropy, and a streaming SHA-256 per split
        file, plus the derived sealed seed. Contains no wall-clock timestamp
        or other non-deterministic field, so it -- like every data file -- is
        bit-identical across repeated calls with the same arguments.

    Determinism: identical (out_dir contents aside, since out_dir must not
    already hold a manifest -- see below) for identical (n_people, seed).

    Memory: O(n_people) auxiliary memory for the train/unseen_person split
    assignment array (a `bool` array, ~1 byte/person -- 4MB at 4M people), but
    O(1) memory for the actual biography/QA text, which is streamed directly
    to disk line by line as `generate_people` yields each person.

    Args:
        out_dir: directory to write into. Created (with parents) if it does
            not exist.
        n_people: number of people in the main population (train + probe +
            unseen_person) and, independently, in the sealed population. Must
            be a non-negative int. `n_people` in {1, 2, 3, 4} is rejected (see
            Raises) because a 90/10 split of that few people leaves one split
            empty, which must never happen silently.
        seed: determinism seed for the main population, its split assignment,
            and its template selection. The sealed population's seed and
            template selection are independently derived from this same
            `seed` (see `_derive_seed`), not equal to it.

    Returns:
        The manifest dict that was also written to `out_dir/manifest.json`.

    Raises:
        TypeError: if n_people or seed is not an int.
        ValueError: if n_people < 0, if seed < 0, or if n_people > 0 but the
            90/10 train/unseen_person split would leave either split empty
            (n_people in {1, 2, 3, 4}).
        FileExistsError: if `out_dir` already contains a `manifest.json` --
            refusing to silently overwrite or interleave with a previous (or
            concurrent) run's output (implementation/03_edge_cases_and_scares.md,
            Dataset generation / Concurrent), or if `out_dir` exists as a
            non-directory file.
    """
    if isinstance(n_people, bool) or not isinstance(n_people, int):
        raise TypeError(f"n_people must be an int, got {type(n_people).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if n_people < 0:
        raise ValueError(f"n_people must be non-negative, got {n_people}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    n_train = round(n_people * _TRAIN_FRACTION)
    n_unseen = n_people - n_train
    if n_people > 0 and (n_train == 0 or n_unseen == 0):
        raise ValueError(
            f"n_people={n_people} is too small for a 90/10 train/unseen_person split "
            f"without leaving a split empty (would produce train={n_train}, "
            f"unseen_person={n_unseen}); use n_people=0 or n_people>=5"
        )

    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"out_dir {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; refusing to overwrite or interleave with a "
            f"previous dataset in {out_dir}. Remove it (or use a fresh out_dir) to regenerate."
        )

    split_rng = np.random.default_rng(_derive_seed(seed, "split"))
    train_mask = np.zeros(n_people, dtype=bool)
    if n_people > 0:
        perm = split_rng.permutation(n_people)
        train_mask[perm[:n_train]] = True

    train_path = out_dir / "train.jsonl"
    probe_path = out_dir / "probe.jsonl"
    unseen_path = out_dir / "unseen_person.jsonl"
    sealed_path = out_dir / "sealed.jsonl"

    text_tmpl_rng = np.random.default_rng(_derive_seed(seed, "train_template"))
    probe_tmpl_rng = np.random.default_rng(_derive_seed(seed, "probe_template"))
    unseen_tmpl_rng = np.random.default_rng(_derive_seed(seed, "unseen_template"))

    n_train_people = 0
    n_probe_lines = 0
    n_unseen_people = 0
    n_unseen_lines = 0

    with train_path.open("w", encoding="utf-8") as train_f, \
            probe_path.open("w", encoding="utf-8") as probe_f, \
            unseen_path.open("w", encoding="utf-8") as unseen_f:
        for i, person in enumerate(generate_people(n_people, seed)):
            if train_mask[i]:
                sentences = render_training_text(person, text_tmpl_rng)
                train_f.write(json.dumps({"person_id": person.person_id, "text": " ".join(sentences)}) + "\n")
                n_train_people += 1
                for attr in ATTRIBUTES:
                    t_idx = int(probe_tmpl_rng.integers(0, len(_PROBE_TEMPLATES[attr])))
                    question, answer = render_probe_qa(person, attr, t_idx)
                    probe_f.write(json.dumps({
                        "person_id": person.person_id, "attribute": attr,
                        "template_idx": t_idx, "question": question, "answer": answer,
                    }) + "\n")
                    n_probe_lines += 1
            else:
                n_unseen_people += 1
                for attr in ATTRIBUTES:
                    t_idx = int(unseen_tmpl_rng.integers(0, len(_PROBE_TEMPLATES[attr])))
                    question, answer = render_probe_qa(person, attr, t_idx)
                    unseen_f.write(json.dumps({
                        "person_id": person.person_id, "attribute": attr,
                        "template_idx": t_idx, "question": question, "answer": answer,
                    }) + "\n")
                    n_unseen_lines += 1

    sealed_seed = _derive_seed(seed, "sealed_population")
    sealed_tmpl_rng = np.random.default_rng(_derive_seed(seed, "sealed_template"))
    n_sealed_people = 0
    n_sealed_lines = 0
    with sealed_path.open("w", encoding="utf-8") as sealed_f:
        for person in generate_people(n_people, sealed_seed):
            n_sealed_people += 1
            for attr in ATTRIBUTES:
                t_idx = int(sealed_tmpl_rng.integers(0, len(_PROBE_TEMPLATES[attr])))
                question, answer = render_probe_qa(person, attr, t_idx)
                sealed_f.write(json.dumps({
                    "person_id": person.person_id, "attribute": attr,
                    "template_idx": t_idx, "question": question, "answer": answer,
                }) + "\n")
                n_sealed_lines += 1

    manifest = {
        "seed": seed,
        "sealed_seed": sealed_seed,
        "n_people_requested": n_people,
        "train_fraction": _TRAIN_FRACTION,
        "bits_per_person": BITS_PER_PERSON,
        "attribute_cardinality": dict(ATTRIBUTE_CARDINALITY),
        "splits": {
            "train": {
                "path": train_path.name,
                "n_people": n_train_people,
                "n_lines": n_train_people,
                "entropy_bits": dataset_entropy_bits(n_train_people),
                "sha256": _sha256_file(train_path),
            },
            "probe": {
                "path": probe_path.name,
                "n_people": n_train_people,
                "n_lines": n_probe_lines,
                "sha256": _sha256_file(probe_path),
            },
            "unseen_person": {
                "path": unseen_path.name,
                "n_people": n_unseen_people,
                "n_lines": n_unseen_lines,
                "entropy_bits": dataset_entropy_bits(n_unseen_people),
                "sha256": _sha256_file(unseen_path),
            },
            "sealed": {
                "path": sealed_path.name,
                "n_people": n_sealed_people,
                "n_lines": n_sealed_lines,
                "entropy_bits": dataset_entropy_bits(n_sealed_people),
                "sha256": _sha256_file(sealed_path),
                "seed": sealed_seed,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
