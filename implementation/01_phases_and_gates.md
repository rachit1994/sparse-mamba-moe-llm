# 01 — Phases and Gates

> Each phase has a **deliverable**, an **acceptance test**, and a **kill condition** with a number.
> A phase is not complete because code exists; it is complete when its acceptance test has been
> independently re-run by the lead and survived attack.
>
> Research-level gates (G0–G7) live in
> [`../initial_research/07_kill_switches.md`](../initial_research/07_kill_switches.md). This file is
> the *build* sequence beneath them.

---

## Phase ladder

```
P0  harness correctness        (container)   ── gate: controls fail when they should
P1  baseline capacity          (Mac mini)    ── gate: reproduce 2.0 bits/param ±25%
P2  dynamic arm capacity       (Mac mini)    ── gate: THE MEASUREMENT
P3  composition / Tier C       (Mac mini)    ── gate: pattern interaction beats chance
P4  scale-up or stop           (Mac mini)    ── gate: G5 U-curve
```

**Phases are strictly ordered.** P2's number is meaningless if P1 did not reproduce the baseline,
because a harness that cannot recover a known result cannot be trusted on an unknown one.

---

## P0 — Harness correctness (container)

**Deliverable:** dataset generator, bits metric, baseline model, training loop — all deterministic,
all with working controls.

**Acceptance:**
- [ ] All three mandatory controls demonstrated *failing when broken* ([02 §2](02_testing_philosophy.md))
- [ ] Determinism verified by re-run, fixed thread count
- [ ] Overfit test: model drives loss ≈ 0 on 20 examples
- [ ] Resume produces bit-identical weights
- [ ] `bits_recovered ≤ entropy` assertion live
- [ ] Analytic parameter count == actual, ≥3 configs
- [ ] Null model bits ≈ 0; `unseen_person` bits ≈ 0

**Kill:** none. P0 cannot fail the research — only the code. If it will not go green honestly,
the harness is wrong and must be fixed before anything else runs.

**Lead verification (not the agent's word):** re-run at a different seed; independently break one
control and confirm it fires; hand-inspect 10 generated people.

---

## P1 — Baseline capacity (Mac mini)

**Deliverable:** dense transformer capacity curve — bits recovered vs N, at ~1M / 10M / 100M params.

**Acceptance:**
- [ ] Curve **plateaus** (saturation reached, not a straight line — S4)
- [ ] Plateau ÷ parameter count reproduces **2.0 bits/param within ±25%** (1.5–2.5)
- [ ] Runs at fp16 or int8, **not ternary** (H5 — ternary makes >2.0 arithmetically impossible)
- [ ] Tier A vs Tier B gap reported (memorisation vs extractable)

**Kill condition — G1:**

| Measured baseline | Action |
|---|---|
| 1.5–2.5 bits/param | **Proceed.** Harness reproduces a known published result |
| 1.0–1.5 or 2.5–3.5 | Investigate before proceeding; likely a convention mismatch (S3) |
| < 1.0 or > 3.5 | **STOP.** The harness does not reproduce a known result; nothing measured after this is trustworthy |

**This is the cheapest kill signal in the project.** If a published, well-replicated number cannot be
reproduced, the measurement apparatus is broken and no novel result from it means anything.

---

## P2 — Dynamic arm capacity (Mac mini) — the measurement

**Deliverable:** the same capacity curve for a dynamic substrate (BDH-style Hebbian or TTT-style
learned state), matched to P1 on parameter count, tokens, dataset, seed, optimiser.

**Acceptance:**
- [ ] Matched on all five axes ([02 §6](02_testing_philosophy.md)); config diff is architecture-only
- [ ] Both arms saturate
- [ ] If the dynamic arm needed different hyperparameters, the **same sweep was run on both arms**

**Kill condition — this is G3, the project's critical gate:**

| Dynamic arm | Verdict | Action |
|---|---|---|
| **> 2.5 b/param** | **PREMISE PROVEN** | Publish. Proceed to P3/P4 |
| 2.0–2.5 | Promising, within noise | Repeat at 3 seeds before claiming anything |
| 1.5–2.0 | Parity | Premise not proven. Decide on throughput/interpretability instead |
| 1.0–1.5 | Worse for knowledge | Keep dynamics for reasoning only; knowledge stays in weights |
| **< 1.0** | **KILL** | Dynamics is worse than plain weights. Revert to the memory-bank design in [09](../initial_research/09_method_comparison_and_decision.md) |

**Parity is a real, reportable outcome.** It is also the most likely one. Deciding that now is what
prevents the result from being tortured into significance later (R4).

---

## P3 — Composition (Mac mini)

**Deliverable:** Tier C results — questions requiring two retrievals and a join, over facts never
co-stated in training ([04 §6](04_golden_dataset.md)).

**Why it matters:** a lookup table scores at chance on Tier C **by construction**. This is where the
original proposal's Component 5 (pattern interaction) becomes measurable rather than rhetorical.

**Acceptance:**
- [ ] Chance level computed analytically per question type, not assumed
- [ ] Both arms evaluated on identical questions
- [ ] Null model scores at chance

**Kill:**

| Result | Action |
|---|---|
| Dynamic arm > dense arm on Tier C | **Strongest possible evidence for the premise** — stronger than raw capacity |
| Both at chance | Composition is not happening in either. Component 5 stays unimplemented; report it |
| Both above chance, equal | Composition is a property of scale, not of dynamics |

---

## P4 — Scale-up or stop

Governed by **G5** (U-curve) in
[`../initial_research/07_kill_switches.md`](../initial_research/07_kill_switches.md).

Only reached if P2 shows promise. If bits/param stops rising as the substrate grows, **stop and ship
the smaller system.** The commitment to do that is made now, in writing, before the data exists.

---

## Quota and interruption policy

Long runs will be interrupted — by session limits, by the machine, by the human.

- Every training script is **resumable**, verified by a bit-identical resume test (H4).
- Every phase writes a `reports/` entry **when it completes**, not at the end of the project.
- Work is committed in **small commits**, pushed as it lands. Nothing valuable lives only in a
  container.
- On quota exhaustion: checkpoint, commit, push, record state, resume on reset. Do not attempt to
  finish a phase by cutting its controls.

---

## Reporting to the human

Reports go to [`../reports/`](../reports/) and are pushed to `main`. One per phase, plus one on any
kill-gate trigger.

Each report states: **what was measured, the actual number, how it could still be wrong, and the
decision taken.** Not a status update — a result with a verdict.

The human is looped in on reports only. Intermediate build progress stays in the repo.
