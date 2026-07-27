# 02 — Testing Philosophy: Green Does Not Mean Working

> Read before writing any test. The single most likely way this project fails is not a bug — it is a
> measurement that looks correct and is not.

---

## 1. The specific danger here

This project's headline output is **one number**: knowledge bits per parameter. A single scalar.

Scalars are trivially faked. Every one of these produces a beautiful, wrong number:

| Failure | What you see | What is actually true |
|---|---|---|
| Test facts leaked into training | 4.0 bits/param | The model memorised the answer key |
| Metric reads the data, not the model | 4.0 bits/param | A random model would score the same |
| Entropy of the dataset miscomputed | 4.0 bits/param | Denominator is wrong; ratio is meaningless |
| Parameter count excludes embeddings | 4.0 bits/param | Different denominator than the baseline |
| Evaluated on training distribution | 4.0 bits/param | No generalisation measured |
| Tokeniser gives away the answer | 4.0 bits/param | Answer inferable from token statistics |

**Every one of these fails silently and looks like success.** No exception is raised. Tests pass.

Therefore: **a test that only passes when the code works is worth very little. You need a test that
fails when the code is subtly wrong.**

---

## 2. The three mandatory controls

No knowledge measurement is reportable without all three. These are cheap; there is no excuse.

### C1 — Null model

Run the identical measurement pipeline on a **randomly-initialised, untrained** model.

- **Expected:** bits recovered ≈ 0 (within noise of chance).
- **If the null model scores meaningfully above zero, your metric is measuring the harness.** Stop.
  Do not report the trained number. Find the leak.

This catches: metric bugs, answer leakage through the prompt, degenerate scoring, chance-level
miscalibration.

### C2 — Mutation

Deliberately break something that must matter, and confirm the number moves in the right direction.

| Mutation | Expected effect |
|---|---|
| Shuffle the mapping between people and attributes in training data | bits → ~0 |
| Zero out the middle third of layers | large drop |
| Train for 1 step instead of N | bits → ~0 |
| Evaluate person IDs never seen in training | bits → ~0 (this is memorisation, correctly) |

**If a mutation does not move the number, the number does not depend on the thing you mutated.**
That is a broken measurement, not a robust one.

### C3 — Contamination proof

State, in the report, **why the model cannot have seen the answers.** For synthetic data this is by
construction (facts are generated from a seeded RNG and exist nowhere else). Say so explicitly and
name the seed. For any real-world data, report n-gram overlap with the training corpus.

---

## 3. Test taxonomy — what each layer is for

| Layer | Answers | Runs where | Trust |
|---|---|---|---|
| **Unit** | does this function do what it says? | container | low — necessary, not sufficient |
| **Property** | do invariants hold for arbitrary inputs? | container | medium |
| **Determinism** | same seed → identical output? | container | high — catches silent nondeterminism |
| **Control (C1–C3)** | does the metric fail when it should? | container | **highest — this is the real test** |
| **Hardware** | throughput, memory, wall-clock | **Mac mini only** | N/A in container |

**Unit tests are the floor, not the bar.** A PR with 100% unit coverage and no controls is not
tested.

### Property tests worth having here

- Dataset entropy computed two independent ways agrees to <0.1%
- Serialising then deserialising a dataset is the identity
- Parameter count from the model equals parameter count from the config
- Bits recovered ≤ dataset entropy (a violation means the metric is broken)
- Bits recovered ≥ 0 for every attribute (negatives mean sign/log-base error)

The fourth one is the most valuable single assertion in this project: **you cannot recover more
information than exists.** If that fires, something is deeply wrong and it is better to know loudly.

---

## 4. Determinism is a correctness property

Every script takes `--seed`. Same seed must produce bit-identical output.

```python
def set_determinism(seed: int) -> None:
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

**Test it:** run twice, hash the outputs, assert equality. Non-reproducible results are not results,
and nondeterminism here would silently invalidate every A/B comparison between architectures.

---

## 5. Numerical hygiene (where a wrong number will actually come from)

- **Log base.** Bits means log₂. PyTorch cross-entropy returns **nats**. Convert: `bits = nats / ln(2)`.
  A missing conversion gives a **1.44× error** — large enough to fake a positive result and small
  enough to look plausible. This is the single most likely arithmetic error in the project.
- **Reduction.** `F.cross_entropy` defaults to `mean`. You usually want `sum` over answer tokens.
  Mixing them silently divides by sequence length.
- **Masking.** Only score the *answer* tokens. Including prompt tokens inflates or deflates
  depending on prompt length.
- **Parameter counting.** Fix one convention and use it for every arm. State it. Allen-Zhu's
  2.0 bits/param baseline includes all trainable parameters; if you exclude embeddings you must
  exclude them from the baseline too or the comparison is invalid.
- **float32 for the metric.** Accumulate log-probs in float32 even if the model runs in bf16/fp16.

---

## 6. The A/B discipline

Every architecture comparison must be matched on:

1. **Parameter count** (same convention, stated)
2. **Training tokens** (exactly, not approximately)
3. **Dataset and seed** (identical)
4. **Optimiser, LR schedule, batch size** — or an explicit statement of why not

Change **one** thing at a time. If you change the architecture and the learning rate together, the
result is uninterpretable and the compute is wasted.

**If an architecture needs a different LR to train at all, that is a finding — report it, do not
quietly tune one arm.** Tuning one arm and not the other is the most common way A/B comparisons in
ML lie.

---

## 7. What "done" means

A measurement is done when **all** of these hold:

- [ ] Runs from a clean checkout with one documented command
- [ ] Deterministic under a fixed seed (verified by re-running)
- [ ] C1 null model run, reported, collapses as expected
- [ ] C2 at least one mutation run, reported, moves as expected
- [ ] C3 contamination argument stated explicitly
- [ ] Bits ≤ entropy assertion active and passing
- [ ] Parameter-counting convention stated and identical across arms
- [ ] Real command + exit code + output tail pasted
- [ ] Container-only numbers labelled `CONTAINER-ONLY`

Missing any box means keep working — not report with caveats.

---

## 8. How the lead verifies

Assume every reported result will be independently attacked, because it will be:

1. Re-run from scratch at a different seed — does the conclusion survive?
2. Null model re-run independently — does it really collapse?
3. An adversarial mutation you did not anticipate — does the number move?
4. Arithmetic re-derived from raw outputs — does the ratio reproduce?
5. The dataset inspected by hand — are the facts really unguessable?

**Anything that only works at one seed is noise. Anything that survives all five is a result.**
Report numbers you would defend under that process, and flag your own doubts before someone else
finds them — a self-flagged weakness costs nothing; a discovered one costs credibility.
