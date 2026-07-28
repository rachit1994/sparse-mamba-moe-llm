# B2 — Knowledge-Bits-Recovered Metric

**Owns:** `src/metrics/bits.py`, `src/metrics/__init__.py`, `tests/test_metrics_bits.py`
**Must not touch:** `src/data/`, `src/models/`, `initial_research/`
**Read first:** [`../00_START_HERE.md`](../00_START_HERE.md) ·
[`../02_testing_philosophy.md`](../02_testing_philosophy.md) §2, §5 ·
[`../04_golden_dataset.md`](../04_golden_dataset.md) §4 ·
[`../03_edge_cases_and_scares.md`](../03_edge_cases_and_scares.md) S1, S3
**Tier:** Build (Sonnet)

---

## Why this exists

This produces **the single headline number of the entire project**. Correctness dominates every other
concern, including speed, elegance, and generality.

## API

```python
answer_cross_entropy_bits(logits, targets, answer_mask) -> float
bits_recovered_per_attribute(model_ce_bits: dict[str, list[float]]) -> dict[str, float]
bits_per_parameter(total_bits_recovered: float, parameter_count: int) -> float
count_parameters(model, include_embeddings: bool = True) -> int
```

```
bits_recovered(j) = Σ_people [ log₂(V_j) − CE_bits(model, correct_value) ]
bits_per_param    = Σ_j bits_recovered(j) / parameter_count
```

## The bug this file exists to prevent

**`torch.nn.functional.cross_entropy` returns NATS, not bits.**

```python
bits = nats / math.log(2)
```

Omitting this inflates the result by exactly **1.4427×**. A true 1.5 bits/param reads as 2.16 —
crossing the 2.0 publication threshold. **This single bug fabricates the project's headline claim,
raises no exception, and looks entirely plausible.** Convert once, at a single named boundary.

## Other numerical requirements

- Score **only answer tokens**; prompt tokens masked out.
- `reduction="sum"` over answer tokens, never `"mean"` (which silently divides by sequence length).
- Accumulate in **float32** even under bf16/fp16 — float16 loses precision past ~2048 additions.
- Clamp per-attribute contributions at **0**; a model can be worse than chance, but negative "stored
  knowledge" is meaningless and would offset real gains elsewhere.
- **One** parameter-counting convention, stated, with `include_embeddings` explicit. Embeddings can be
  30–50% of a small model — enough to manufacture a 2× difference from nothing (S3).

## Assertions live in the code, not only in tests

1. `bits_recovered ≤ dataset_entropy` — **you cannot recover more information than exists.** The
   highest-value assertion in the project; if it fires, the metric is broken.
2. `bits_recovered ≥ 0` per attribute after clamping.

## Tests — `tests/test_metrics_bits.py`

1. **Chance level**: hand-built logits with `p(correct) = 1/V` → `bits_recovered == 0`.
2. **Perfect model**: `p(correct) ≈ 1.0` → `bits_recovered ≈ log₂(V)` per item.
3. **NATS-vs-BITS regression**: fails if the `/log(2)` is removed. *The most important test in the file.*
4. **Masking**: changing prompt tokens must not change the score; changing answer tokens must.
5. **Entropy bound**: the `≤ entropy` assertion actually raises when violated.
6. **Determinism**: identical inputs → identical float output.

Edge cases: zero answer tokens after masking (**raise** — a silent `0.0` reads as "knows nothing");
`p(correct) = 0` giving `-log₂(0) = inf` (clamp with a documented epsilon); logits/targets shape
mismatch (raise with **both** shapes in the message); the same person scored twice (dedupe or raise —
double-counting inflates linearly).

## Acceptance — the deliverable is evidence the tests work

Break the implementation at least **three** ways, confirm the matching test **fails**, then revert:

| Break | Test that must fail |
|---|---|
| Remove the `/log(2)` conversion | nats-vs-bits regression |
| Drop the answer mask | masking test |
| Switch `sum` to `mean` | perfect-model test |
| Remove the entropy-bound assertion | entropy-bound test |

Report exactly what you broke, the **real failure output**, and that you reverted.
**A test that does not fail when you break its target is broken — fix it and say so.**

## Reporting

Template in [`../00_START_HERE.md`](../00_START_HERE.md) §5. Show the **actual numeric outputs** of the
chance-level and perfect-model tests — those two numbers are how the lead checks the formula by hand.

## Environment

Python 3.11, torch 2.13 CPU, numpy. **Add no dependencies.** Match `schema.py` style: type hints,
docstrings with `Raises:`, typed errors, never return `None` on failure, no bare `except`.
