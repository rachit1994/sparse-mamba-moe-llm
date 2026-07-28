# 03 — Edge Cases and Scares

> Every entry is a way this project produces a **confidently wrong number**. Most fail silently:
> no exception, no red test, a plausible result. Read the rows for your area before you start —
> most bugs you could introduce are already here.
>
> Format: **what breaks · how it lies to you · how to detect it · what to do.**

---

## Tier 1 — Scares that invalidate the headline result

These do not degrade the number. They **invert** it.

### S1. The nats/bits error (1.44×)
- **Breaks:** `F.cross_entropy` returns nats; the metric needs bits.
- **Lies:** inflates bits/param by exactly `1/ln2 = 1.4427`. A true 1.5 reads as 2.16 — crossing the
  2.0 publication threshold. **This single bug fabricates the project's headline claim.**
- **Detect:** dedicated regression test that fails if the conversion is removed
  ([02 §5](02_testing_philosophy.md)).
- **Do:** convert once, at a single named boundary. Never inline `/log(2)` in multiple places.

### S2. Contamination
- **Breaks:** evaluated facts appeared in training.
- **Lies:** unbounded inflation; the model recites rather than stores.
- **Detect:** `unseen_person` split must score ≈ 0. If it does not, there is leakage.
- **Do:** synthetic facts from a private seed. State the construction in every report.

### S3. Parameter-count mismatch between arms
- **Breaks:** arm A counts embeddings, arm B does not; or the baseline uses a different convention
  than Allen-Zhu's 2.0 bits/param.
- **Lies:** a pure denominator artifact reads as an architecture win. Embeddings can be 30–50% of a
  small model — enough to manufacture a 2× difference from nothing.
- **Detect:** analytic parameter count asserted equal to actual, per config; convention printed in
  every report header.
- **Do:** one convention, stated, identical across arms **and** against the external baseline.

### S4. Saturation never reached
- **Breaks:** dataset entropy is below model capacity, so the model learns everything.
- **Lies:** you measure **dataset size**, not model capacity. Bits/param becomes an arbitrary
  function of N.
- **Detect:** sweep N; capacity is the **plateau**. A straight line through the origin means you
  never saturated.
- **Do:** bracket N from 0.25× to 4× of expected capacity ([04 §2](04_golden_dataset.md)).

### S5. Tuning one arm only
- **Breaks:** the dynamic arm needs a different LR to train at all, so it gets tuned; the baseline
  does not.
- **Lies:** reports an architecture difference that is a hyperparameter difference.
- **Detect:** review the config diff between arms. Anything beyond the architecture is suspect.
- **Do:** identical optimiser/schedule/batch, or run the **same sweep on both arms** and report
  best-of-sweep for each. "It needed a different LR" is a finding — report it, do not bury it.

### S6. Peeking at the sealed split
- **Breaks:** sealed data consulted during development.
- **Lies:** stage-to-stage comparisons become tuning-to-test; the U-curve is fiction.
- **Detect:** access logging; sealed set lives behind an explicit function that records every call.
- **Do:** one look per gate. If peeked, regenerate at a new seed and say so.

---

## Tier 2 — Edge cases in code (the ordinary ladder)

### Dataset generation
| Rung | Case | Required behaviour |
|---|---|---|
| Empty | `n_people=0` | valid; entropy 0; no crash |
| Boundary | `n_people=1`; splits rounding to 0 | must not silently produce an empty split — raise |
| Huge | 4M people | streams; constant memory; must not OOM |
| Duplicate | name collision from RNG | detect and either reject or re-draw; **collisions corrupt entropy** |
| Malformed | missing attribute | `Person.__post_init__` raises (already enforced) |
| Wrong type | `n_people` float/negative | typed error, not silent coercion |
| Concurrent | two writers, same out_dir | fail fast on existing manifest; no partial overwrite |

**Name collisions are the sneaky one.** Two people with the same ID means the dataset has less
entropy than `N × BITS_PER_PERSON`, so every bits/param number is inflated. Assert uniqueness.

### Metric
| Rung | Case | Required behaviour |
|---|---|---|
| Empty | zero answer tokens after masking | raise — a silent 0.0 reads as "knows nothing" |
| Boundary | p(correct) exactly 0 | `-log2(0)` = inf; clamp with documented epsilon |
| Negative | model worse than chance | clamp at 0 per attribute, documented |
| Huge | 4M people accumulation | float32 accumulator; **float16 loses precision at ~2048 additions** |
| Malformed | logits/targets shape mismatch | raise with both shapes in the message |
| Duplicate | same person scored twice | dedupe or raise; double-counting inflates linearly |

### Training
| Rung | Case | Required behaviour |
|---|---|---|
| Boundary | 0 steps; 1 step | no crash; loss finite |
| Failure | OOM mid-run | checkpoint survives; resume is bit-identical |
| Concurrent | two runs, same output dir | fail fast, do not interleave |
| Huge | context longer than `block_size` | truncate explicitly and log, never wrap silently |
| Untrusted | NaN/Inf loss | abort immediately with the step number — do not train through it |

---

## Tier 3 — Hardware and environment

### H1. Container numbers reported as results
Timing, throughput, and memory from the Linux container are **meaningless** for an M4 Mac mini
(4 CPU cores, no GPU, no MLX). Label every such number `CONTAINER-ONLY`.
**Correctness generalises from container to Mac mini; performance does not.**

### H2. Thermal throttling
A Mac mini throttles under sustained load. A 60-second benchmark measures the boost clock, not the
sustained one, and training timelines derived from it are optimistic.
**Do:** measure over ≥10 minutes; report sustained, not peak.

### H3. Nondeterminism from thread count
Different thread counts change float reduction order, so results differ in the last bits.
**Do:** pin threads in the harness; hash-compare only under a fixed thread count.

### H4. Checkpoint/resume divergence
Optimiser state, RNG state, and dataloader position must all be restored. Restoring weights alone
gives a run that looks fine and silently differs.
**Do:** the resume test asserts *bit-identical* final weights, not "similar loss".

### H5. Quantization × capacity (G-CAP)
Allen-Zhu's 2.0 bits/param is verified only down to int8. At ternary (1.6 bits/param of storage) the
**storage bound binds before the capacity law** — so beating 2.0 bits/param at ternary is
arithmetically impossible, not merely hard.
**Do:** run capacity comparisons at fp16 or int8. Any ternary capacity claim must be measured
directly and is an upper bound until then. See
[initial_research/10_dynamic_substrate.md](../initial_research/10_dynamic_substrate.md) §7.

---

## Tier 4 — Research-level scares

### R1. Memorisation vs extractable knowledge
A model can store a fact and be unable to answer a rephrased question. Allen-Zhu found paraphrase
diversity materially changes extractable knowledge.
**Detect:** Tier A vs Tier B gap ([04 §6](04_golden_dataset.md)). A large gap means the capacity
number overstates usable knowledge.
**This is a real phenomenon, not a bug — report it rather than tuning it away.**

### R2. The dynamic arm may not train at all
Hebbian/attractor training is historically unstable; the field moved Hopfield → attention for this
reason. BDH claims stability at ≤1B, and that claim is exactly what reproduction must check.
**Do:** if the dynamic arm will not converge, that is a **result** (gate G1). Do not spend weeks
tuning it into looking good.

### R3. BDH's public repo ≠ its headline numbers
The released code is the paper's baseline variant; the 97.4% Sudoku-Extreme figure comes from an
unreleased internal implementation.
**Do:** plan only against the published language-modelling numbers. Never cite 97.4% as reproducible.

### R4. The result may be "no difference"
Parity (≈2.0 bits/param both arms) is the most likely single outcome.
**Do:** decide *now* that parity is a publishable, reportable result. A project that can only report
success will report success.

---

## Detection checklist before any result is reported

- [ ] `unseen_person` scores ≈ 0 (S2)
- [ ] Null model scores ≈ 0 ([02 §2](02_testing_philosophy.md))
- [ ] `bits_recovered ≤ entropy` assertion active and passing (S1, S4)
- [ ] Parameter-count convention stated, identical across arms (S3)
- [ ] N swept; plateau visible, not a straight line (S4)
- [ ] Config diff between arms contains only the architecture (S5)
- [ ] Names verified unique (entropy integrity)
- [ ] Determinism verified by re-run at fixed thread count (H3)
- [ ] Container-only numbers labelled (H1)
- [ ] Precision stated; ternary claims flagged as upper bounds (H5)
