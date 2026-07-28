# B1 — Synthetic Biography Dataset Generator

**Owns:** `src/data/generate.py`, `src/data/__init__.py`, `tests/test_data_generate.py`
**Must not touch:** `src/data/schema.py` (lead-owned), `src/metrics/`, `src/models/`, `initial_research/`
**Read first:** [`../00_START_HERE.md`](../00_START_HERE.md) ·
[`../04_golden_dataset.md`](../04_golden_dataset.md) (your spec) ·
[`../03_edge_cases_and_scares.md`](../03_edge_cases_and_scares.md) S2, S4, and the dataset table

---

## Why this exists

The dataset **is** the experiment. The whole project measures "bits of knowledge stored per
parameter," which requires knowing **exactly how many bits the dataset contains**. Synthetic
biographies are the only construction that gives both exact entropy and a contamination proof.

If the dataset is wrong, every downstream number is wrong and no amount of careful modelling
recovers it.

## API

```python
generate_people(n_people: int, seed: int) -> Iterator[Person]
render_training_text(person: Person, rng) -> list[str]                    # >=5 templates/attribute
render_probe_qa(person: Person, attribute: str, template_idx: int) -> tuple[str, str]
write_dataset(out_dir: Path, n_people: int, seed: int) -> dict            # streams JSONL, returns manifest
```

## Hard requirements

| # | Requirement | Why |
|---|---|---|
| R1 | Names `<Token><Token>-<4 digits>` from a synthetic syllable inventory; **no real person representable** | This is the contamination proof (S2) |
| R2 | Names verified **unique** | A collision means entropy < `N × BITS_PER_PERSON`, silently inflating every bits/param number |
| R3 | Attributes sampled **independently**, near-uniformly, at cardinalities from `schema.py` | Correlation overstates entropy; the model exploits it |
| R4 | Probe uses **held-out templates** never seen in training | Otherwise you measure sentence recall, not knowledge |
| R5 | Templates contain **no attribute-specific tokens** | Template leakage inflates recovery |
| R6 | Splits: train (90% of people), probe (same people, held-out phrasings), `unseen_person` (10%, never trained), sealed (separate seed) | `unseen_person` is the leakage control — it must score ≈ 0 |
| R7 | Deterministic on `--seed`; streams to disk; no OOM at 4M people | Data is never committed; generator + seed is the artifact |

## Tests — `tests/test_data_generate.py`

1. Determinism: same seed twice → identical SHA256.
2. Entropy matches `n × BITS_PER_PERSON` to <0.1%.
3. Attribute-pair mutual information ≈ 0 (state and justify the threshold).
4. Chi-square uniformity per attribute.
5. `train ∩ probe = ∅` on phrasings; `train ∩ unseen_person = ∅` on people.
6. No template contains any attribute value token.
7. Serialise → deserialise is the identity.

Edge cases that must be handled explicitly: `n_people = 0` (valid, entropy 0), `n_people = 1`, splits
rounding to an empty split (**raise**, do not silently produce it), negative or float `n_people`
(typed error, not coercion), existing manifest in `out_dir` (fail fast, no partial overwrite).

## Acceptance — the deliverable is evidence the tests work

Break the implementation at least **three** ways, confirm the matching test **fails**, then revert:

| Break | Test that must fail |
|---|---|
| Make two attributes correlated | mutual-information test |
| Reuse a probe template in training | disjointness test |
| Ignore the seed | determinism test |
| Allow duplicate names | uniqueness / entropy test |

Report exactly what you broke, the **real failure output**, and that you reverted.
**A test that does not fail when you break its target is broken — fix it and say so.**

## Reporting

Template in [`../00_START_HERE.md`](../00_START_HERE.md) §5. Include the **full generated text for one
sample person** so the lead can eyeball it — hand inspection catches things no assertion will.

## Environment

Python 3.11, numpy + stdlib. torch exists but you almost certainly do not need it. **Add no
dependencies.** Match `schema.py` style: type hints, docstrings with `Raises:`, typed errors, no bare
`except`.
