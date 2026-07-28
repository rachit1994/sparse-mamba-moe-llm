# B3 — Baseline Model and Training Loop

**Owns:** `src/models/`, `src/train.py`, `tests/test_models_*.py`
**Must not touch:** `src/data/`, `src/metrics/`, `initial_research/`, `src/data/schema.py`
**Read first:** [`../00_START_HERE.md`](../00_START_HERE.md) ·
[`../02_testing_philosophy.md`](../02_testing_philosophy.md) §4, §6 ·
[`../03_edge_cases_and_scares.md`](../03_edge_cases_and_scares.md) S3, S5, H3, H4

---

## Why this exists

We measure how many bits of knowledge a model stores per parameter. The published dense-transformer
baseline is **2.0 bits/param** (Allen-Zhu & Li). **This arm is the control** — a plain, correct,
boring transformer whose job is to reproduce that number.

A later arm tests a dynamic/Hebbian substrate against it. Both arms must be matched exactly on
parameter count, training tokens, dataset, seed, and optimiser. **If this arm is wrong, the
comparison is worthless and so is the project.**

## Deliverables

```
src/models/config.py    dataclass: n_layer, n_head, d_model, vocab_size, block_size, tie_embeddings
src/models/tiny_lm.py   decoder-only transformer; parameter count exactly predictable from config
src/train.py            training loop, CLI, deterministic, checkpointing, resumable
```

## Hard requirements

| # | Requirement | Why |
|---|---|---|
| R1 | Parameter count **analytically computable from config without instantiating the model**, and asserted equal to the actual count | Parameter count is the denominator of the headline metric (S3) |
| R2 | Configs for ~1M, ~10M, ~100M parameters | Capacity sweep needs a range |
| R3 | `--seed` makes runs **bit-identical**; verified by re-running and hashing weights | Non-reproducible results are not results |
| R4 | Checkpoint + resume produces **bit-identical** final weights vs an uninterrupted run | Real runs are long and get interrupted (H4) |
| R5 | Logits hook exposing `(logits, targets, answer_mask)` with documented shapes | The metric agent consumes this |
| R6 | NaN/Inf loss aborts immediately with the step number | Never train through a diverged run |
| R7 | No network at training time | Contamination risk; will not exist on target hardware |
| R8 | Default test configs are **tiny** (seconds, not minutes) | 4 CPU cores, no GPU in the container |

R4 is subtle: resuming must restore **optimiser state, RNG state, and dataloader position**, not just
weights. Restoring weights alone produces a run that looks fine and silently differs.

## Tests — `tests/test_models_tiny_lm.py`

1. **Analytic == actual parameter count**, for ≥3 configs.
2. **Determinism**: same seed twice → identical weight hashes (pin thread count, see H3).
3. **Resume correctness**: 10 steps vs 5 + checkpoint + resume 5 → identical final weights.
4. **Overfit**: model drives loss ≈ 0 on 20 examples.
5. **Logits hook contract**: shapes and dtypes.

**Test 4 is the most important.** It is the one thing that catches "the training loop silently does
nothing" — the single most common and most embarrassing failure in ML code. It must actually fail if
the optimiser step were removed. **Verify that by removing it.**

## Acceptance — the deliverable is evidence the tests work

Break the implementation at least **three** ways, confirm the matching test **fails**, then revert:

| Break | Test that must fail |
|---|---|
| Remove `optimizer.step()` | overfit test |
| Drop positional embeddings | overfit test (loss plateaus higher) |
| Restore weights but not optimiser state on resume | resume test |
| Omit an embedding matrix from the analytic count | param-count test |

Report exactly what you broke, the **real failure output**, and that you reverted.
**A test that does not fail when you break its target is broken — fix it and say so.**

## Reporting

Use the template in [`../00_START_HERE.md`](../00_START_HERE.md) §5. Include the **actual loss curve**
from the overfit test.

Label any timing or throughput number `CONTAINER-ONLY` — this container is 4 CPU cores with no GPU
and no MLX, and such numbers are not valid results for the M4 Mac mini target (H1).

## Environment

Python 3.11, torch 2.13 CPU, numpy. **Add no dependencies** — everything must run on macOS/MLX later.
Match `src/data/schema.py` style: type hints, docstrings with `Raises:`, typed errors, no bare
`except`, no returning `None` on failure.
