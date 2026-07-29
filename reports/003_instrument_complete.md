# 003 — Measuring Instrument Complete — **the experiment's subject is not built**

Date: 2026-07-29 · Commit: `8e42045`

---

## Verdict

**The instrument is finished and verified. The thing it is meant to measure does not exist yet.**

That distinction is the whole report. Everything built so far measures a *conventional transformer* —
which is the control arm. The dynamic substrate that the thesis is actually about has not been built,
and neither has the structured dataset on which the thesis can win.

## Numbers: what is verified

Full suite, every file, single clean run:

```
182 passed in 208.81s
```

| Component | Verification | Number |
|---|---|---|
| **Parameter count** | agreement across 4 independent sources, incl. a formula derived from GPT-2 structure without reading the code | `101,558,272` exact on all 4 |
| **Bits metric** | vs pure-NumPy reference on 6 non-saturated mixed-probability cases | agreement **< 1e-6** |
| | accumulation drift over 5000 items | **3.1e-07** relative |
| | deleting the nats→bits conversion | **6 tests fail** |
| **Tokenizer** | OOV over 1,500 training lines + all probe QA | **0** |
| | name consistency, 200 names × 8 real contexts | **0 inconsistent** |
| | probe answers recovered exactly by answer mask | **250/250** |
| | vocabulary | **2,818** |
| **P0 controls** | null model | **0.00 bits** (0.0% of ceiling) |
| | trained, seen people | **2036.24 bits** (99.9%) |
| | trained, unseen people | **11.32 bits** (0.6%) |
| | entropy bound | 2036.24 ≤ 2038.39 ✓ |
| **Plateau detector** | 6 known-answer curves + 5 malformed inputs | battery **PASSED** |
| | extrapolation vs analytic ground truth | predicted **9.98%**, exact **10.0%** |
| **Saturation guard** | straight line | raises `UnsaturatedCurveError` |
| | saturated curve | emits **37.0136 bits/param** |

## The information-theoretic framing

Restated without reference to language models:

```
eta := knowledge_bits / substrate_bits        eta <= 1.0   (Shannon)

fp16    eta = 0.125      8.0x headroom
int8    eta = 0.250      4.0x headroom   <- best measured
int4    eta = 0.175      5.7x headroom
```

**Transformers store knowledge at ¼ of the information-theoretic bound.** The thesis asks whether a
dynamical encoding closes that 4× gap. Nothing in that question mentions tokens or attention.

Note the inversion the parameter-denominated view was hiding: **ternary has the lowest bits/param and
the highest η.** Denominating in parameters reverses the ranking of encoding schemes.

## Feasibility, measured not assumed

Full 1000-exposure saturating runs, pessimistic efficiency band:

| model | params | N to saturate | tokens | days |
|---|---:|---:|---:|---:|
| tiny | 27,008 | 2,120 | 148 M | **0.05** |
| 100k | 100,000 | 7,848 | 549 M | **0.67** |
| 1m | 1,057,152 | 82,963 | 5,807 M | 74.8 |
| 10m | 10,844,160 | 851,023 | 59,572 M | 7,873 |

Measured peak matmul: **114 GFLOP/s** on 4 threads. Reporting that as *training* throughput would have
overstated feasibility ~10×, so the estimate carries an explicit 5–20% efficiency band.

## What is NOT built — and this is the point

| Missing | Why it matters |
|---|---|
| **Dataset B (latent structure)** | On the flat dataset, Shannon **forbids** any encoding from beating another. Both a pattern-former and a lookup table must tie. The thesis literally cannot win there. |
| **The dynamic substrate arm** | Knowledge must be able to enter via test-time state or Hebbian change, **not** SGD into base weights. Otherwise it is a transformer with extra steps and the thesis is untested. |

Two agent attempts at Dataset B were killed by session limits before producing code.

## Deliberate descoping of P1

P1 originally required reproducing Allen-Zhu's 2.0 bits/param. On reflection that is **not on the
critical path**: both arms are measured by the same instrument on the same data, so the **ratio
between arms is valid even if the absolute calibration is off**. Calibration matters only for absolute
claims.

At 75 days for the 1M point, letting it gate the experiment would be straying. It runs later as an
anchor, not as a blocker.

## Mistakes: 15 logged, 7 severity-1

S1 means *would have produced a wrong headline number, silently, with no failing test*.

The two most instructive:

**M5 — the instrument was built around the rival hypothesis.** Parameter-denominated metric,
SGD-into-weights training, and a dataset with no latent structure. Three of four core choices favoured
the thing the thesis argues against.

**M15 — near-miss on false corroboration.** I was about to cite the sweep's plateau detector agreeing
with mine as independent evidence. It had **read my file and adopted my criterion**, and said so in its
own comments. Shared criterion means shared failure mode. What actually validates both is the
ground-truth battery, where asymptotes are known analytically — stronger than implementation agreement
would ever have been.

## Code removed

`src/integration.py`, **611 lines**: zero tests, unverified, and returned `0.0` bits identically at 0,
200 and 800 training steps — unable to distinguish a trained model from an untrained one. Replaced by
the verified 139-line path.

## Open risk

The sweep rejects any capacity curve containing a **negative interval gain**, arguing a decrease
signals training instability rather than saturation. That is stricter than my detector and rejected my
noisy battery case (one interval at −31.17 under 2% noise). Erring strict is right — refusing to report
beats reporting garbage — but it is **brittle**: a real sweep with ordinary measurement noise will hit
a negative interval and hard-fail. Watch this on the first real sweep.

## Next

1. **Dataset B** — latent archetypes, exact entropy, so structure exists to discover.
2. **Dynamic substrate arm** — Hebbian/fast-weight, knowledge entering without gradient steps on base
   weights.
3. Sweep both arms across the structure axis. **The claim under test is a slope**, not a scalar:
   η_dynamic should rise with latent structure while η_static stays flat.
