"""Tests for the synthetic biography dataset generator (src/data/generate.py).

Read implementation/02_testing_philosophy.md before extending this file: a test
that only passes when the code works is worth very little -- every test here is
designed to FAIL if the specific defect it targets is present. See the task
report for the "deliberate breakage" verification log (real captured pytest
output for each of the four required breaks, plus a byte-identical revert
confirmation) -- that log is the actual evidence these tests work, not just
that they pass.

Test sizes are kept small (at most 100,000 synthetic people, well under a
second each to generate) so the whole suite runs in well under a minute on 4
CPU cores. The 4M-people / no-OOM claim (R7) is checked via a memory-bounded
streaming test and an algebraic name-space-capacity check, NOT by actually
generating 4M rows here.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import resource
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from src.data.generate import (
    _CITY_VALUES,
    _EMPLOYER_VALUES,
    _MAJOR_VALUES,
    _MONTH_NAMES,
    _NAME_DOMAIN_SIZE,
    _PROBE_TEMPLATES,
    _TRAIN_TEMPLATES,
    _UNIVERSITY_VALUES,
    _YEAR_MIN,
    _YEARS_SPAN,
    _derive_name_key_and_attr_seeds,
    _derive_seed,
    _feistel_permute,
    _iter_people,
    _make_person_id,
    generate_people,
    render_probe_qa,
    render_training_text,
    write_dataset,
)
from src.data.schema import ATTRIBUTE_CARDINALITY, ATTRIBUTES, Person

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_written_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """One write_dataset() call, reused read-only across several tests."""
    out_dir = tmp_path_factory.mktemp("small_dataset")
    manifest = write_dataset(out_dir, n_people=4_000, seed=99)
    return out_dir, manifest


@pytest.fixture(scope="module")
def medium_people() -> list[Person]:
    """20,000 people, big enough for chi-square / mutual-information estimation
    (min per-cell expected count >= 5 for the largest-cardinality attributes)
    while staying fast (~0.2s to generate)."""
    return list(generate_people(20_000, seed=4242))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ===========================================================================
# The 7 required tests (implementation/06_agent_briefs/B1_dataset_generator.md)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Determinism: same seed twice -> identical SHA256.
# ---------------------------------------------------------------------------


def test_determinism_same_seed_twice_identical_sha256(tmp_path: Path) -> None:
    manifest_a = write_dataset(tmp_path / "run_a", n_people=2_000, seed=555)
    manifest_b = write_dataset(tmp_path / "run_b", n_people=2_000, seed=555)

    for split in ("train", "probe", "unseen_person", "sealed"):
        hash_a = manifest_a["splits"][split]["sha256"]
        hash_b = manifest_b["splits"][split]["sha256"]
        print(f"[determinism] {split}: a={hash_a} b={hash_b}")
        assert hash_a == hash_b

    # No wall-clock or other nondeterministic field in the manifest -> the
    # manifest.json files themselves must be byte-identical, not just equal
    # split hashes.
    text_a = (tmp_path / "run_a" / "manifest.json").read_text(encoding="utf-8")
    text_b = (tmp_path / "run_b" / "manifest.json").read_text(encoding="utf-8")
    assert text_a == text_b


def test_different_seeds_produce_different_output(tmp_path: Path) -> None:
    """Companion sanity check: hashes are not trivially constant."""
    manifest_a = write_dataset(tmp_path / "seed_a", n_people=2_000, seed=1)
    manifest_b = write_dataset(tmp_path / "seed_b", n_people=2_000, seed=2)
    assert manifest_a["splits"]["train"]["sha256"] != manifest_b["splits"]["train"]["sha256"]


# ---------------------------------------------------------------------------
# 2. Entropy matches n x BITS_PER_PERSON to <0.1%, computed two independent ways.
# ---------------------------------------------------------------------------


def test_entropy_matches_analytic_formula_independently_recomputed(
    small_written_dataset: tuple[Path, dict],
) -> None:
    """Cross-checks generate.py's reported entropy_bits against a SEPARATE
    computation path: raw math.log2 over schema.ATTRIBUTE_CARDINALITY, not
    schema.BITS_PER_PERSON or schema.dataset_entropy_bits (which generate.py
    itself also calls, so reusing them here would not be independent).
    """
    _, manifest = small_written_dataset
    independent_bits_per_person = sum(math.log2(v) for v in ATTRIBUTE_CARDINALITY.values())

    for split in ("train", "unseen_person", "sealed"):
        n = manifest["splits"][split]["n_people"]
        expected = n * independent_bits_per_person
        actual = manifest["splits"][split]["entropy_bits"]
        rel_err = abs(actual - expected) / expected if expected else abs(actual - expected)
        print(f"[entropy] {split}: n={n} expected={expected!r} actual={actual!r} rel_err={rel_err:.2e}")
        assert rel_err < 0.001, f"{split}: relative error {rel_err} >= 0.1%"


def test_entropy_zero_people_is_zero(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path / "zero_entropy", n_people=0, seed=1)
    assert manifest["splits"]["train"]["entropy_bits"] == 0.0
    assert manifest["splits"]["unseen_person"]["entropy_bits"] == 0.0
    assert manifest["splits"]["sealed"]["entropy_bits"] == 0.0


# ---------------------------------------------------------------------------
# 3. Attribute-pair mutual information ~= 0 (threshold stated and justified).
# ---------------------------------------------------------------------------

_MI_BINS: Final[int] = 10
# Bias-corrected (Miller-Madow-style) plug-in MI is centred near 0 bits under
# true independence; the theoretical finite-sample bias at N=20,000, k=10 is
# (k-1)^2 / (2 N ln2) ~= 0.0029 bits (see _mi_bits docstring). This threshold
# is ~7x that bias -- a generous margin against sampling noise across 10
# pairs, while still 1-2 orders of magnitude below the MI a real correlation
# produces (verified: injecting `uni_idx = city_idx % n_uni` in generate.py
# and rerunning this test gives MI > 1 bit for that pair; see task report).
_MI_THRESHOLD_BITS: Final[float] = 0.02

_VALUE_TO_INDEX: Final[dict[str, dict[str, int]]] = {
    "birth_city": {v: i for i, v in enumerate(_CITY_VALUES)},
    "university": {v: i for i, v in enumerate(_UNIVERSITY_VALUES)},
    "major": {v: i for i, v in enumerate(_MAJOR_VALUES)},
    "employer": {v: i for i, v in enumerate(_EMPLOYER_VALUES)},
}


def _bucket_index(attribute: str, value: str, k: int = _MI_BINS) -> int:
    """Map a raw attribute value to one of `k` bins.

    A deterministic function of one attribute's independently-drawn raw value
    is still independent of every other attribute's raw value, so testing MI
    between bucketed attributes tests the same independence property as the
    un-bucketed values -- just at a resolution the sample size can actually
    estimate. A full 73,000 x 1,000 birth_date x birth_city contingency table
    would need tens of millions of samples before the naive MI estimator's
    bias becomes small relative to any real signal; a 10x10 table does not.
    """
    if attribute == "birth_date":
        year = int(value.rsplit(" ", 1)[-1])
        return year % k
    return _VALUE_TO_INDEX[attribute][value] % k


def _mi_bits(x_bins: np.ndarray, y_bins: np.ndarray, k: int = _MI_BINS) -> float:
    """Bias-corrected plug-in mutual information estimate, in bits.

    mi_naive = sum_{x,y} p(x,y) * log2( p(x,y) / (p(x) p(y)) )

    Subtracts the standard first-order finite-sample bias of the naive
    plug-in MI estimator under the null of independence (Miller 1955):
    (kx-1)(ky-1) / (2 N ln2). Clamped at 0 (a bias-corrected estimate can go
    slightly negative under true independence; negative "correlation" is not
    meaningful here, we only care how far above 0 it lands).
    """
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


def test_attribute_pair_mutual_information_near_zero(medium_people: list[Person]) -> None:
    bucketed = {
        attr: np.array([_bucket_index(attr, p.attributes[attr]) for p in medium_people])
        for attr in ATTRIBUTES
    }
    for a, b in itertools.combinations(ATTRIBUTES, 2):
        mi = _mi_bits(bucketed[a], bucketed[b])
        print(f"[MI] {a} x {b}: {mi:.5f} bits (threshold {_MI_THRESHOLD_BITS})")
        assert mi < _MI_THRESHOLD_BITS, f"{a} x {b}: MI={mi} bits >= threshold"


# ---------------------------------------------------------------------------
# 4. Chi-square uniformity per attribute.
# ---------------------------------------------------------------------------

# Strict significance level: only fail if uniformity is rejected at p < 0.1%.
# Computed via the Wilson-Hilferty cube-root normal approximation to the
# chi-square CDF (numpy + stdlib `math.erfc` only -- no scipy, per the "no new
# dependencies" constraint). Accurate for df >= ~10; all df used here are >=99.
_CHI_SQUARE_ALPHA: Final[float] = 0.001


def _chi_square_uniform_pvalue(counts: np.ndarray) -> float:
    k = len(counts)
    n = float(counts.sum())
    expected = n / k
    stat = float(((counts - expected) ** 2 / expected).sum())
    df = k - 1
    mean = 1 - 2 / (9 * df)
    std = math.sqrt(2 / (9 * df))
    z = ((stat / df) ** (1 / 3) - mean) / std
    return 0.5 * math.erfc(z / math.sqrt(2))  # upper-tail p-value


@pytest.mark.parametrize("attribute", ["major", "university", "birth_city", "employer"])
def test_attribute_chi_square_uniformity(medium_people: list[Person], attribute: str) -> None:
    vocab_size = ATTRIBUTE_CARDINALITY[attribute]
    counts = np.zeros(vocab_size)
    lookup = _VALUE_TO_INDEX[attribute]
    for p in medium_people:
        counts[lookup[p.attributes[attribute]]] += 1
    p_value = _chi_square_uniform_pvalue(counts)
    print(f"[chi2] {attribute}: min_count={counts.min():.0f} max_count={counts.max():.0f} p={p_value:.4f}")
    assert p_value > _CHI_SQUARE_ALPHA, f"{attribute}: uniformity rejected, p={p_value}"


def test_birth_year_chi_square_uniformity(medium_people: list[Person]) -> None:
    counts = np.zeros(_YEARS_SPAN)
    for p in medium_people:
        year = int(p.attributes["birth_date"].rsplit(" ", 1)[-1])
        counts[year - _YEAR_MIN] += 1
    p_value = _chi_square_uniform_pvalue(counts)
    print(f"[chi2] birth_year: min_count={counts.min():.0f} max_count={counts.max():.0f} p={p_value:.4f}")
    assert p_value > _CHI_SQUARE_ALPHA


# ---------------------------------------------------------------------------
# 5. train n probe = empty on phrasings; train n unseen_person = empty on people.
# ---------------------------------------------------------------------------


def test_train_probe_phrasing_and_train_unseen_person_disjoint(
    small_written_dataset: tuple[Path, dict],
) -> None:
    out_dir, _ = small_written_dataset
    train_records = _read_jsonl(out_dir / "train.jsonl")
    unseen_records = _read_jsonl(out_dir / "unseen_person.jsonl")
    sealed_records = _read_jsonl(out_dir / "sealed.jsonl")

    # -- people: train and unseen_person must never name the same person --
    train_people = {r["person_id"] for r in train_records}
    unseen_people = {r["person_id"] for r in unseen_records}
    sealed_people = {r["person_id"] for r in sealed_records}
    assert train_people.isdisjoint(unseen_people)
    # Beyond the brief's literal requirement: also verify the sealed
    # population (a real latent collision risk found and fixed during this
    # review -- see write_dataset's "Name collision" docstring paragraph).
    assert train_people.isdisjoint(sealed_people)
    assert unseen_people.isdisjoint(sealed_people)

    # -- phrasings, static template pools: TRAIN and PROBE pools per attribute
    #    must never share a phrasing (also asserted at import time; re-checked
    #    here against the generator's actual output, not just its constants).
    for attr in ATTRIBUTES:
        assert set(_TRAIN_TEMPLATES[attr]).isdisjoint(_PROBE_TEMPLATES[attr])

    # -- phrasings, generated output: no training paragraph contains the
    #    literal rendered form of ANY probe-pool template for that same
    #    person. Checks the FULL probe template pool per attribute (not just
    #    whichever template_idx probe.jsonl happened to sample for that
    #    person), and checks substring containment rather than splitting the
    #    paragraph into "sentences" by punctuation -- an earlier version of
    #    this test split on ". " and silently missed a reused probe template
    #    ending in "?" (verified: it passed when it should have failed; see
    #    task report for the real failing-then-fixed transcript).
    for r in train_records:
        person_id = r["person_id"]
        text = r["text"]
        for attr in ATTRIBUTES:
            for template in _PROBE_TEMPLATES[attr]:
                rendered = template.format(name=person_id)
                assert rendered not in text, (
                    f"train text for {person_id!r} contains probe-pool phrasing for "
                    f"{attr!r}: {rendered!r}"
                )


# ---------------------------------------------------------------------------
# 6. No template contains any attribute value token.
# ---------------------------------------------------------------------------


def test_no_template_contains_attribute_value_token() -> None:
    all_value_words: set[str] = set()
    for bank in (_CITY_VALUES, _UNIVERSITY_VALUES, _MAJOR_VALUES, _EMPLOYER_VALUES):
        for value in bank:
            all_value_words.update(w.lower() for w in value.split())
    all_value_words.update(m.lower() for m in _MONTH_NAMES)

    leaks: list[tuple[str, str, str]] = []
    for pool in (_TRAIN_TEMPLATES, _PROBE_TEMPLATES):
        for attr, templates in pool.items():
            for template in templates:
                literal = template.replace("{name}", " ").replace("{value}", " ")
                for word in re.findall(r"[A-Za-z']+", literal):
                    if word.lower() in all_value_words:
                        leaks.append((attr, template, word))

    print(f"[leak-check] checked {len(all_value_words)} distinct value words across all templates")
    assert leaks == [], f"template(s) leak an attribute-value token: {leaks}"


# ---------------------------------------------------------------------------
# 7. Serialise -> deserialise is the identity.
# ---------------------------------------------------------------------------


def test_serialize_deserialize_round_trip_is_identity(
    small_written_dataset: tuple[Path, dict],
) -> None:
    out_dir, manifest = small_written_dataset
    seed = manifest["seed"]
    n_people = manifest["n_people_requested"]

    people_by_id = {p.person_id: p for p in generate_people(n_people, seed)}

    probe_records = _read_jsonl(out_dir / "probe.jsonl")
    assert len(probe_records) > 0
    for rec in probe_records[:500]:
        # JSON round trip is exact.
        assert json.loads(json.dumps(rec)) == rec
        # Deserialised record regenerates the exact same (question, answer)
        # from the generator -- not just "json parses", but "the write was
        # a faithful, reproducible encoding of the generator's output".
        person = people_by_id[rec["person_id"]]
        question, answer = render_probe_qa(person, rec["attribute"], rec["template_idx"])
        assert question == rec["question"]
        assert answer == rec["answer"]


# ===========================================================================
# Uniqueness (R2) -- dedicated test, separate from the entropy test, since
# entropy_bits is computed from a requested COUNT and would not by itself
# notice duplicate identities within that count.
# ===========================================================================


def test_generated_names_are_unique_at_scale() -> None:
    n = 100_000
    ids = [p.person_id for p in generate_people(n, seed=17)]
    unique = set(ids)
    print(f"[uniqueness] requested={n} unique={len(unique)}")
    assert len(unique) == n


def test_main_and_sealed_populations_share_name_key_no_collision_at_scale() -> None:
    """Dedicated regression test for a real bug found and fixed during this
    review: two INDEPENDENTLY-keyed Feistel permutations of the same ~3.3B
    name domain are NOT collision-free at realistic scale (birthday bound).
    `write_dataset`'s fix makes the sealed population reuse the main
    population's name_key at a disjoint index offset instead.

    Exercises the exact private mechanism `write_dataset` uses (bypassing its
    JSONL/file-I/O, which would make an n=100,000 check too slow for this
    suite -- ~9.8s measured for a full `write_dataset(100_000, ...)` call vs
    ~1.7s here) at a scale where the OLD (buggy) design reliably produced
    collisions in practice: 2 observed empirically while diagnosing this bug
    at n=100,000 each (the birthday-bound expectation is
    n_main * n_sealed / domain_size ~= 3.0); the small_written_dataset
    fixture's n=4,000 is FAR too small to reliably exercise this (expected
    overlap there is ~0.005, i.e. it would very likely show 0 collisions even
    under the old buggy code -- this is why this separate, larger-scale test
    exists rather than relying solely on the disjointness test above).
    """
    n = 100_000
    seed = 777

    main_name_key, main_attr_seeds = _derive_name_key_and_attr_seeds(seed)
    sealed_seed = _derive_seed(seed, "sealed_population")
    _, sealed_attr_seeds = _derive_name_key_and_attr_seeds(sealed_seed)

    main_ids = {p.person_id for p in _iter_people(n, main_name_key, main_attr_seeds, index_start=0)}
    sealed_ids = {p.person_id for p in _iter_people(n, main_name_key, sealed_attr_seeds, index_start=n)}

    assert len(main_ids) == n
    assert len(sealed_ids) == n
    assert main_ids.isdisjoint(sealed_ids)


def test_feistel_permute_is_bijective_exhaustive() -> None:
    """Exhaustive (not sampled) bijectivity check on a domain small enough to
    fully enumerate: every one of the `domain_size` inputs must map to a
    distinct output in [0, domain_size). This is the property the whole
    collision-free-by-construction design (R2) rests on.
    """
    domain_size = 5_000
    key = 0xABCDEF
    outputs = [_feistel_permute(i, domain_size, key) for i in range(domain_size)]
    assert len(set(outputs)) == domain_size
    assert all(0 <= o < domain_size for o in outputs)


def test_make_person_id_matches_regex_shape() -> None:
    pattern = re.compile(r"^[A-Za-z]+ [A-Za-z]+-\d{4}$")
    for i in range(200):
        person_id = _make_person_id(i, name_key=123)
        assert pattern.match(person_id), person_id


# ===========================================================================
# R7 / "no OOM at 4M people": memory-bounded, not an actual 4M-row run.
# ===========================================================================


def test_streaming_memory_is_bounded_not_linear_in_n() -> None:
    """Peak RSS growth while iterating N people (each discarded immediately,
    only a running count kept) should not scale with N: `_iter_people`
    batches attribute sampling in fixed-size (`_GEN_CHUNK`) chunks rather than
    materialising the whole population.

    Uses `resource.getrusage(...).ru_maxrss` (a single syscall reading the
    kernel's tracked peak RSS) rather than `tracemalloc` (which instruments
    every allocation and made this single test take >25s at these sizes --
    an unacceptable fraction of the "well under a minute" budget for the
    negligible benefit of per-allocation attribution here).

    `ru_maxrss` is monotonic non-decreasing for the process's lifetime, so
    the second delta (5,000 -> 200,000 people, 40x more) isolates the
    incremental growth attributable to processing 40x more people: if memory
    were O(n), this delta would be tens of MB; observed O(1) design gives
    ~0.1MB.
    """
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    count = sum(1 for _ in generate_people(5_000, seed=31337))
    assert count == 5_000
    after_small_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    count = sum(1 for _ in generate_people(200_000, seed=31337))  # 40x more people
    assert count == 200_000
    after_large_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    delta_kb = after_large_kb - after_small_kb
    print(
        f"[memory] baseline={baseline_kb}KB after_5,000={after_small_kb}KB "
        f"after_200,000={after_large_kb}KB incremental_delta_for_40x_more_people={delta_kb}KB"
    )
    # Generous absolute bound: O(1) design predicts ~0KB extra; a genuine
    # O(n) leak at this scale would add tens of MB, not fit under 50MB.
    assert delta_kb < 50_000


def test_name_domain_capacity_covers_4m_people_twice_over() -> None:
    """write_dataset needs 2 x n_people identities (main + sealed) from the
    shared Feistel domain; confirm it comfortably covers 2 x 4,000,000 without
    actually generating that many rows."""
    assert 2 * 4_000_000 <= _NAME_DOMAIN_SIZE


# ===========================================================================
# Edge cases (implementation/03_edge_cases_and_scares.md, Dataset generation)
# ===========================================================================


def test_n_people_zero_is_valid_no_crash(tmp_path: Path) -> None:
    assert list(generate_people(0, seed=1)) == []
    manifest = write_dataset(tmp_path / "zero", n_people=0, seed=1)
    assert manifest["splits"]["train"]["n_people"] == 0
    assert manifest["splits"]["unseen_person"]["n_people"] == 0
    assert manifest["splits"]["sealed"]["n_people"] == 0


def test_n_people_one_generates_one_person() -> None:
    people = list(generate_people(1, seed=1))
    assert len(people) == 1
    assert isinstance(people[0], Person)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_split_rounding_to_empty_split_raises(tmp_path: Path, n: int) -> None:
    with pytest.raises(ValueError, match="too small"):
        write_dataset(tmp_path / f"n{n}", n_people=n, seed=1)


def test_n_people_five_is_smallest_valid_split(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path / "n5", n_people=5, seed=1)
    assert manifest["splits"]["train"]["n_people"] > 0
    assert manifest["splits"]["unseen_person"]["n_people"] > 0


@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_n_people_raises_value_error(bad: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_people(bad, seed=1)
    with pytest.raises(ValueError):
        write_dataset(tmp_path / "neg", bad, seed=1)


@pytest.mark.parametrize("bad", [3.5, "100", None, True, False])
def test_non_int_n_people_raises_type_error(bad: object, tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        generate_people(bad, seed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        write_dataset(tmp_path / "badtype", bad, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_seed", [-1, 3.5, "1", None, True])
def test_bad_seed_raises_typed_error(bad_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_people(10, seed=bad_seed)  # type: ignore[arg-type]


def test_existing_manifest_fails_fast_no_partial_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "concurrent"
    write_dataset(out, n_people=50, seed=1)
    before = (out / "train.jsonl").read_bytes()

    with pytest.raises(FileExistsError):
        write_dataset(out, n_people=999, seed=2)  # different args must not clobber

    after = (out / "train.jsonl").read_bytes()
    assert before == after


def test_out_dir_exists_as_non_directory_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "not_a_dir"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_dataset(p, n_people=10, seed=1)


def test_missing_attribute_person_rejected_by_schema() -> None:
    """R7/schema interaction: Person.__post_init__ already enforces this
    (schema.py, read-only); confirmed reachable from this module's imports."""
    with pytest.raises(ValueError, match="missing attributes"):
        Person(person_id="X Y-0001", attributes={a: "v" for a in ATTRIBUTES[:-1]})


# ---------------------------------------------------------------------------
# render_training_text / render_probe_qa: floor unit coverage.
# ---------------------------------------------------------------------------


def test_render_training_text_one_sentence_per_attribute() -> None:
    person = next(iter(generate_people(1, seed=1)))
    rng = np.random.default_rng(0)
    sentences = render_training_text(person, rng)
    assert len(sentences) == len(ATTRIBUTES)
    for s in sentences:
        assert person.person_id in s


def test_render_probe_qa_unknown_attribute_raises() -> None:
    person = next(iter(generate_people(1, seed=1)))
    with pytest.raises(ValueError, match="unknown attribute"):
        render_probe_qa(person, "not_a_real_attribute", 0)


def test_render_probe_qa_template_idx_out_of_range_raises() -> None:
    person = next(iter(generate_people(1, seed=1)))
    with pytest.raises(ValueError, match="out of range"):
        render_probe_qa(person, "major", 999)


def test_render_probe_qa_non_int_template_idx_raises_type_error() -> None:
    person = next(iter(generate_people(1, seed=1)))
    with pytest.raises(TypeError):
        render_probe_qa(person, "major", 1.0)  # type: ignore[arg-type]
