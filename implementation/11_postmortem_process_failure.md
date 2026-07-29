# 11 — Post-Mortem: Why a Planned Project Still Drifted

> Written by the lead about the lead. The plan was good. It was not used as a gate, and that is a
> different failure from not having a plan.

---

## The number

| | Tokens |
|---|---:|
| All delegated builds | 1,219,310 |
| **Rework attributable to spec drift** | **~714,867 (59%)** |
| Dispatches killed by quota with zero output | 5 |

Rework, itemised:

- `dynamic.py` built as neither TTT nor BDH → **311,301** (M17)
- Dataset B forced afterwards, because the flat-only design cannot test the thesis → **203,566** (M5)
- `integration.py`: 611 lines, zero tests, deleted → **~200,000**

**The waste was building the wrong thing, not over-explaining it.** Token discipline aimed at
verbosity would have saved a rounding error. Token discipline aimed at *spec fidelity* would have
saved 59%.

---

## Root cause: the brief paraphrased the plan instead of quoting it

`04_architecture.md` §3.1 said:

> **Option A — BDH (Hebbian synaptic state)** … **Option B — TTT (state as a learned model).**
> Selection criterion: bits-of-knowledge per parameter … Not aesthetics.

The dispatch brief I actually wrote said:

> SUGGESTED MECHANISM (evaluate on merits): A Hebbian / fast-weight associative memory …
> This is Hopfield/fast-weight territory.

**The words "TTT" and "BDH" do not appear in that brief.** I paraphrased two named published
architectures into a generic mechanism family, and the agent built exactly what I asked for.

The agent did nothing wrong. **M17 was introduced in my brief, before a line of code existed.** No
amount of output review catches a defect injected upstream of the output.

### Why I paraphrased

Hebbian outer-product is easy to specify and easy to build. TTT requires an inner-loop gradient;
BDH requires neuron-synapse dynamics with emergent attention. I selected for **buildable** and told
myself it was **representative**. That is the single most junior thing in this whole record, and it
happened at the exact moment the plan was supposed to constrain me.

---

## Four contributing failures

**1. No traceability requirement.** No brief was required to name the doc section and paper equation
it implements. Had `B-dynamic` carried `implements: 10_dynamic_substrate.md §3.1 (BDH, arXiv:2509.26507)`,
"Hebbian associative memory" would have been visibly non-compliant on sight.

**2. Sequenced by tooling convenience, not dependency.** The pattern encoder is Components 1–2 — the
foundation. I built the metric, then a transformer, then memory, then a substrate, and only built the
encoder when the user demanded it. Why? The metric harness needed *a model to measure*, and a
transformer was available. **I built what the tooling wanted rather than what the thesis required.**

**3. I verified correctness and never verified faithfulness.** Every component I adjudicated was
internally correct: the metric matched an independent reference to 1e-6, parameter counts agreed four
ways, the entropy formula was exact. None of that asks *is this the thing the plan specified?*
Two different gates. I only ever ran one.

**4. I did the work I could do instead of the work only I could do.** I hand-wrote the tokenizer, the
plateau detector, and verification scripts — Sonnet work. Meanwhile the irreplaceable lead task
(*does this brief faithfully encode the plan?*) went undone. Being busy is not the same as leading.

---

## What the process should have been

| Stage | Rule | Would have caught |
|---|---|---|
| **1. Spec freeze** | Before any brief: a one-page mechanism spec per component, **quoting the paper's equations verbatim**, with the paper ID | M17 |
| **2. Brief traceability** | Mandatory field: `implements: <doc §> / <paper eq>`. A brief that cannot cite one does not get dispatched | M17 |
| **3. Lead reviews the brief** | The brief is the artifact most worth reviewing — it is upstream of everything | M17, M5 |
| **4. Dependency order** | Build in thesis order (encoder → patterns → interaction), never tooling order | pattern encoder built last |
| **5. Faithfulness gate before correctness gate** | Ask "is this what the plan specified?" *before* "is this internally right?" | M5, M17 |
| **6. Lead writes specs, not code** | Any lead-authored source file needs a written justification (silent-corruption trap only) | ~4 files |

**Stage 5 is the missing gate.** Everything else in this project has a gate — saturation, entropy
bound, null model, leakage, exposure count. There was no gate on *did we build the thing we said we
would build*, which is why two S1 defects walked straight through a heavily-gated process.

---

## What was NOT the problem

Worth stating so the fix targets the right thing:

- **Not the research.** `initial_research/` 01–10 held up. The capacity math, the prior-art map, the
  BDH/TTT identification — all correct and all still standing.
- **Not the testing philosophy.** Break-and-revert, null controls, independent reimplementation, and
  ground-truth validation all worked, and caught real defects (the 1.44× nats bug, the name-collision
  bug, the MI estimator being 4× off).
- **Not the agents.** Sonnet and Haiku built what they were told. The one agent error worth logging
  (M9, zero-agreement-as-evidence) was a reasoning slip I had made twice myself.
- **Not verbosity.** See the numbers above.

The plan was sound and the verification was sound. **The link between them — brief fidelity — was
unguarded.**

---

## Standing rules from here

1. **No dispatch without a frozen spec** that quotes the paper.
2. **Every brief cites `implements:`.** No citation, no dispatch.
3. **Faithfulness is checked before correctness**, on the brief and on the result.
4. **The lead writes specs and gates. Sonnet writes code.** Lead-authored source requires a written
   exception.
5. **Build in dependency order**, and when tooling pressure suggests otherwise, that pressure is the
   signal to stop.
