# 002 — P0 Controls — **PASS. The apparatus works.**

Date: 2026-07-28 · Phase: P0 · Commit: `1c1b55b`
Supersedes the blocking finding in [001](001_P0_harness.md).

---

## The numbers

Measured end to end through an independently-wired pipeline. **CONTAINER-ONLY.**

```
entropy ceiling per split :   2038.39 bits
NULL model                :      0.00 bits    (  0.0% of ceiling)
TRAINED, seen people      :   2036.24 bits    ( 99.9% of ceiling)
TRAINED, unseen people    :     11.32 bits    (  0.6% of ceiling)
```

| P0 gate | Threshold | Result |
|---|---|---|
| Null model ≈ 0 | ≤ 5% of ceiling | **0.0%** — PASS |
| Unseen-person ≈ 0 (leakage) | ≤ 5% of ceiling | **0.6%** — PASS |
| Trained ≫ null (detection) | ≥ 10× | **180×** — PASS |
| `bits_recovered ≤ entropy` | hard bound | **2036.24 ≤ 2038.39** — PASS |

## Verdict

**PROCEED to P1.** The measuring apparatus reads the model, not the harness.

## Why the null-model zero is not a vacuous pass

This is the part that matters, and it is where a careless reading goes wrong.

A null model scoring 0.00 could mean two very different things: *the metric works*,
or *the per-item clamp is zeroing everything regardless of input*. The second would
be a metric that always reports zero — passing the control while being incapable of
measuring anything.

**The trained number is what disambiguates it.** 2036.24 bits on the same pipeline,
same clamp, same code path. A metric that always returns zero cannot produce that.
The pair together is the evidence; neither number alone is.

## Why unseen people score 0.6% rather than exactly 0

Honest reading: this is **not** fact leakage. The model learns answer *formatting*
and the *value distributions* — that birth cities come from a particular vocabulary,
that dates have a particular shape. That beats chance slightly on people it has never
seen, without knowing anything about those specific people.

The discriminating figure is the **180× gap** between seen (99.9%) and unseen (0.6%).
Genuine leakage would narrow that gap, not preserve it.

## How this was verified

The pipeline used here was wired by hand from the public component APIs and
**deliberately does not import `src/metrics/bits.py`** — it recomputes cross-entropy
from log-softmax independently. This is the same two-implementations discipline that
caught earlier problems in the metric. An agent-built pipeline is landing separately;
agreement between the two will be evidence rather than a shared bug.

Training curve (400 steps, answer-token loss in nats): `0.1885 → 0.0113 → 0.0053 → 0.0032`.

## What this is NOT

**This is not a capacity result, and must not be quoted as one.**

- 40 people = 2,038 bits sits far inside the model's capacity; **saturation is not
  reached**, so the ratio measures dataset size, not model capacity (edge case S4).
- The **~1000-exposure regime** that the 2.0 bits/param baseline requires was not used.
- Container-only: 4 CPU cores, no GPU, no MLX. The M4 Mac mini has run nothing.

99.9% recovery here means "a 400k-parameter model memorised 2 kbit," which is
unremarkable and expected. The interesting number is the **plateau** as N grows, and
that is P1.

## How this could still be wrong

- **Only one seed.** The controls have not been repeated across seeds; a
  seed-specific artifact would not yet be visible.
- **Tiny scale.** 40 people, 2-layer model, 64-dim. Behaviour at 10M+ parameters and
  100k+ people is untested and is where saturation effects live.
- **Probe-format training.** I trained directly on probe Q/A, which measures
  memorisation-and-recall. P1 must train on declarative text and probe with held-out
  phrasings — the harder and more meaningful test (Tier A vs Tier B).
- **No Mac mini number exists.** Every hardware assumption in
  [`../initial_research/01_feasibility.md`](../initial_research/01_feasibility.md)
  remains unmeasured.

## Decision

P0 is closed. P1 proceeds under three conditions that are now blocking, not advisory:

1. **int8 or fp16 — never int4 or ternary.** int4 measures 0.7 bits/param, so it
   holds twice the parameters of int8 but *less* knowledge. Beating 2.0 at ternary is
   arithmetically impossible.
2. **~1000 exposures per fact.** At ~100 exposures the ceiling is 1.0 bits/param, and
   P1 would fail its gate for reasons unrelated to architecture.
3. **Sweep N until the curve plateaus.** A straight line through the origin means
   saturation was never reached and the number is meaningless.

Before P1 runs on the Mac mini, `bench/` (gate G0) must be executed there — every
throughput figure in the research is still a projection from vendor specs.
