# 07 — Kill Switches

> A gate that cannot fail is not a gate. Each of these has a **number**, a **measurement**, and a
> **decision**. Cost of a wrong "continue" is months; cost of a wrong "stop" is an unbuilt idea.
> These are written to be *easy to fail*.

**Rule: no gate may be softened after seeing its result.** If a threshold looks wrong, that is an
argument to be made *before* the measurement, in writing, with a reason.

---

## G0 — Hardware reality check (before any code)

**Measure** with `bench/`, on the actual Mac mini.

| # | Assumption | Threshold | If it fails |
|---|---|---|---|
| G0.1 | Decode-effective bandwidth ≥ 70% of 120 GB/s | ≥ 84 GB/s | Re-derive all throughput; halve tok/s targets |
| G0.2 | Sustained GPU MFU ≥ 30% (MLX) | ≥ 1.32 TFLOP/s | Training timelines scale inversely — recheck [05](05_training_and_population.md) |
| G0.3 | SSD sequential read ≥ 2.5 GB/s | ≥ 2.5 GB/s | Block fetch degrades; Tier B may be unviable |
| G0.4 | SSD random 4 KB read ≥ 50K IOPS | ≥ 50K | **Design already de-risked** — hash addressing is flat to 20K IOPS |
| G0.5 | `iogpu.wired_limit_mb` raisable to ~12 GiB, stable | no swap thrash under 1 h load | Cut expert pool from 24B to 12B |
| G0.6 | Ternary kernels ≥ 50% of fp16 bandwidth efficiency on Metal | ≥ 50% | Fall back to INT4; storage doubles to 400 GB at S4 |

**Kill condition:** if G0.1 **and** G0.2 both fail by >2×, the machine is not the machine described
in [01](01_feasibility.md) and every downstream number is void. Stop and re-plan.

G0.4 is deliberately lenient because [09 §3](09_method_comparison_and_decision.md) chose an
architecture that is insensitive to it — that was the point of the choice.

---

## G1 — Reproduce a published result (before building anything new)

**The cheapest possible kill signal.** Do not build a novel system before reproducing a known one.

| Measurement | Threshold | Decision |
|---|---|---|
| Reproduce Meta/DeepSeek memory-layer gains on factual QA at small scale (≤500M base) | ≥ 50% of published relative gain | Continue |
| | 10–50% | Investigate before continuing |
| | < 10% | **STOP.** If a published result won't reproduce here, nothing novel will |

**Timebox: 3 weeks.** If reproduction is not achieved in 3 weeks, that is itself a signal about
tooling maturity on this platform.

---

## G2 — Quantization survives (before committing to ternary)

*Scaling Laws for Precision* predicts PTQ degradation **grows** with pretraining tokens — and every
candidate checkpoint is heavily trained.

| Measurement | Threshold | Decision |
|---|---|---|
| Perplexity delta, ternary vs fp16, on the chosen backbone | < 5% | Continue with ternary |
| | 5–15% | Use natively-ternary BitNet b1.58 2B4T only; abandon post-hoc ternarization |
| | > 15% | **Fall back to INT4.** Storage at S4 doubles to 400 GB ⇒ S4 needs external NVMe regardless |

---

## G3 — Written memory works as well as trained memory (**the core research risk**)

Every published memory result *trained* its values. This project *writes* them. That substitution is
what makes the project possible ([05 §3](05_training_and_population.md)) and is the least-validated
claim in it.

| Measurement | Threshold | Decision |
|---|---|---|
| Written-memory accuracy vs trained-memory accuracy, matched size, small scale | ≥ 90% of trained | Continue — the whole plan is viable |
| | 70–90% | Continue at reduced ambition; cap at S2 |
| | < 70% | **STOP the write-path.** Only trained memory works, and training it costs 10⁵ years ⇒ **the trillion-parameter target is dead.** Fall back to a ~30B trainable system |

**This is the single most important gate in the document.** Everything above 30B parameters depends
on it. Run it at S1, on the smallest possible configuration, as early as possible.

---

## G4 — The memory is actually load-bearing

| Measurement | Threshold | Decision |
|---|---|---|
| Ablate memory at inference (Arm D, [06](06_evaluation.md)) | ≥ 10% degradation | Memory is used |
| | 3–10% | Memory is marginal — investigate routing/addressing |
| | < 3% | **STOP.** The memory is decorative; the system is a small dense model with expensive dead weight |

Paired check — **the honest-opponent gate**:

| Measurement | Threshold | Decision |
|---|---|---|
| vs. dense + conventional RAG over the same corpus (Arm E) | beats RAG on ≥ 3 of 5 metric families | Continue |
| | ties RAG | **STOP.** This is an expensive reimplementation of retrieval — use RAG |

---

## G5 — The U-curve (the gate on "one trillion")

Three independent results say total sparse parameters is the worst axis to scale
([09 §6](09_method_comparison_and_decision.md)). This gate makes 1T an earned outcome.

Fit accuracy-per-active-parameter across stages:

| Observation | Decision |
|---|---|
| Still rising S1 → S2 → S3 | **Build S4.** The trillion is justified by measurement |
| Flattens at S2 or S3 | **Stop at that stage.** Ship it; the remaining parameters buy nothing |
| Turns over (accuracy *falls*) | **Stop immediately and roll back.** More memory is actively harmful |

**Explicit commitment: if the curve flattens at S2 (126B), the project ships a 126B system and the
trillion-parameter goal is formally abandoned.** Writing this down now is what makes it possible to
honour later.

---

## G6 — Usability

A system nobody can use is not a result.

| Measurement | Threshold | Decision |
|---|---|---|
| Decode throughput | ≥ 15 tok/s | Continue |
| | 5–15 tok/s | Usable only for batch/agentic work — narrow the product claim |
| | < 5 tok/s | **STOP.** This is the Kimi-K2-port failure mode ([01 §5](01_feasibility.md)) |
| Prefill, 2048-token prompt | ≤ 10 s | Continue |
| | > 30 s | Interactive use is dead; batch only |
| Sustained 1-hour run | no thermal throttle > 20%, no swap | Continue |

---

## G7 — Scope discipline (a gate on the researcher, not the system)

The proposal has 8 components. Three (Pattern Interaction, Pattern Synthesis, Reasoning Engine) have
**no validated implementation at any scale** in the surveyed literature.

| Rule | Enforcement |
|---|---|
| No work on Components 5/7 until G3, G4, G5 all pass | They are the least-grounded and most seductive parts |
| Component 7 (Pattern Synthesis) stays **deferred** | No surveyed work does this; it is a research programme in itself |
| Every component ships with a falsifiable metric or is marked **deferred**, never "implemented" | [06 §6](06_evaluation.md) |

---

## Summary — the decision tree

```
G0 hardware ──fail──► re-plan; numbers are void
  │pass
G1 reproduce ──fail──► STOP (nothing novel will work if the known doesn't)
  │pass
G2 quantization ──fail──► INT4 fallback, S4 needs external NVMe
  │pass
G3 written≈trained ──fail──► STOP the trillion; fall back to ~30B trainable
  │pass                       ◄── THE CRITICAL GATE
G4 memory load-bearing ──fail──► STOP (decorative memory / RAG does it cheaper)
  │pass
G5 U-curve ──flat──► ship at that stage, abandon 1T on evidence
  │rising
G6 usability ──fail──► batch-only, or STOP below 5 tok/s
  │pass
  ▼
Build S4 (1T). Earned, not assumed.
```

---

## What would make this project a success even if it fails

Worth stating, because it changes how failure should be handled:

- **G1 failing** produces a documented account of why memory layers don't reproduce on Apple
  silicon — useful to everyone doing on-device work.
- **G3 failing** produces the first measurement of the written-vs-trained memory gap. Nobody has
  published that number. It is a genuine contribution.
- **G5 turning over** locates the U-curve's optimum empirically — DeepSeek asserts the curve exists
  but the turning point is unpublished.

**Three of the six gates produce a publishable negative result when they fail.** Design the
experiments so the negative is recorded and reported, not discarded.
