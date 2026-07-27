# 06 — Evaluating Intelligence: How to Compare Without Fooling Ourselves

> The original proposal asks "how would we compare intelligence" without defining it. This document
> defines it, then builds a protocol that can return a **negative** result.

---

## 1. The definition being used

Chollet, *On the Measure of Intelligence* (2019):

> The intelligence of a system is a measure of its **skill-acquisition efficiency** over a scope of
> tasks, with respect to **priors, experience, and generalization difficulty**.

The sharp consequence, in Chollet's words: *unlimited priors or unlimited training data allow
experimenters to "buy" arbitrary levels of skill in a way that masks the system's own generalization
power.*

**This lands directly on this project.** A system whose knowledge sits in a 974B-parameter written
memory bank has enormous *experience* by construction. Raw benchmark scores will therefore
**overstate** its intelligence relative to a dense model of equal active compute. Any evaluation that
does not control for this is measuring the size of the memory bank, not the quality of the
architecture.

---

## 2. The three questions that must be answered separately

Conflating these is the main way this project could fool itself.

| Question | Metric | Honest expectation |
|---|---|---|
| **Q1. Is knowledge in memory competitive with knowledge in weights?** | factual QA at matched active compute | **Yes** — this is the well-supported part |
| **Q2. Does memory help reasoning, or only recall?** | reasoning benchmarks at matched active compute | **Uncertain** — Engram says yes (BBH +5.0), Meta says gains concentrate on factual tasks |
| **Q3. Does the system acquire new skills efficiently?** | ARC-AGI-2, few-shot adaptation | **Probably not** — this is the frontier claim and the most likely failure |

---

## 3. The controlled comparison

**Every comparison must be matched on active compute, not on total parameters.** Comparing a 126B-total
system to a 126B dense model is meaningless; the whole thesis is that total parameters are the wrong
axis.

| Arm | Purpose |
|---|---|
| **A. Dense baseline** — 2.6B dense, same tokens | the number to beat |
| **B. MoE baseline** — 2.6B active, 26B total | isolates the value of sparsity alone |
| **C. Memory system** — 2.6B active, 126B total | isolates the value of memory over sparsity |
| **D. Memory, ablated** — C with memory zeroed at inference | proves the memory is actually used |
| **E. RAG baseline** — A + a conventional retrieval index over the same corpus | **the honest opponent** |

**Arm E is the one that matters and the one most likely to be skipped.** If a 2.6B dense model with
an off-the-shelf RAG pipeline over the same corpus matches the memory system, the entire architecture
is an expensive reimplementation of retrieval. That comparison must be run first, not last.

**Arm D is the integrity check.** If zeroing the memory does not degrade performance, the memory is
decorative — a real failure mode for sparsely-accessed parameters.

---

## 4. Benchmark suite

### Tier 1 — knowledge (expected strength)
NaturalQuestions, TriviaQA, MMLU, CMMLU. Meta reported **>100%** relative gains on the first two from
memory layers; Engram reported MMLU +3.4 / CMMLU +4.0 over iso-param, iso-FLOP MoE. **These are the
numbers to reproduce first**, because failing to reproduce a published result is the cheapest possible
kill signal.

### Tier 2 — reasoning (the contested claim)
BBH, ARC-Challenge, GSM8K, MATH, HumanEval. Engram's gains here (BBH +5.0, ARC-C +3.7, HumanEval
+3.0, MATH +2.4) **exceeded** its knowledge gains. If this reproduces, the "memory buys knowledge but
not reasoning" concern is retired. If it does not, the architecture is a knowledge store and should
be described as one.

### Tier 3 — long context (the mechanism check)
Multi-Query NIAH. Engram reported **84.2 → 97.0**, attributed to lookup handling local dependencies
and freeing attention for global context. This is a *mechanistic* prediction — if the mechanism is
real, this metric moves; if the gains come from somewhere else, it won't. **This is the most
diagnostic single benchmark in the suite.**

### Tier 4 — skill acquisition (the actual intelligence claim)
**ARC-AGI-2.** Current landscape (July 2026): frontier systems ~92.5%, average individual human 66%,
human test panel 60%. The compute-capped Kaggle track — no internet, ~$0.20/task — scored **24%** in
2025.

**The compute-capped track is the correct comparison for this project**, not the frontier leaderboard.
A Mac mini at ~30 W for 120 s costs ~$0.0002 of electricity — three orders of magnitude *inside* the
$0.20/task budget. **Beating 24% under those constraints would be a genuinely publishable result.**
Matching the frontier's 92.5% is not a realistic target and should not be set as one.

### Tier 5 — continual learning (the differentiating claim)
The proposal's distinctive promise is *learning without retraining*. Protocol: inject new facts into
memory **without gradient updates**, then measure (a) recall of the new fact, (b) **retention of old
facts** (catastrophic forgetting), (c) whether the fact composes with existing knowledge rather than
being merely retrievable. No standard benchmark exists; this needs a purpose-built probe set.

---

## 5. Efficiency metrics — where this architecture should actually win

Report these alongside every accuracy number. If the system is not winning here, it has no reason to
exist.

| Metric | Why |
|---|---|
| **Accuracy per active parameter** | the thesis, stated numerically |
| **Accuracy per joule** | BitNet: 0.028 J/token vs 0.186–0.649 J for fp16 peers |
| **Accuracy per GB of RAM** | the actual constraint on this machine |
| **tok/s at fixed accuracy** | UltraMem's claim: 2–6× faster than MoE at parity |
| **Bytes read per token** | the binding physical constraint ([01](01_feasibility.md)) |

**Accuracy per active parameter is the headline metric for this project.** RETRO's 25× (7.5B matching
GPT-3 175B) is the number to measure against.

---

## 6. Making "pattern formation" falsifiable

Components 2, 5 and 7 of the original proposal (Pattern Space, Pattern Interaction, Pattern
Synthesis) are currently **unmeasurable as written** — which means they cannot fail, which means they
cannot be research. The only concrete instrument found:

**Train a sparse autoencoder on the backbone's residual stream** and count recoverable
monosemantic features. Anthropic extracted 1M / 4M / 34M interpretable features from Claude 3 Sonnet
this way, showing features are multilingual, multimodal, abstract, and **causally steerable**.

Operationalised:

- **Pattern Space:** number of monosemantic features recoverable per unit of backbone parameter.
  Claim: memory-augmented backbones support *more* features per parameter than dense ones.
- **Pattern Interaction:** do features compose? Steer feature A and feature B simultaneously; test
  whether the model behaves as if the conjunction is represented.
- **Pattern Synthesis:** do *new* features appear after memory population that were absent before,
  without any weight update? This is the sharpest possible test of the proposal's central claim, and
  it is a clean experiment.

Without this instrumentation those three components should be marked **deferred**, not implemented.

---

## 7. Ways this evaluation could lie, and the countermeasure

| Failure | Countermeasure |
|---|---|
| Memory bank contains benchmark test data | Decontaminate the corpus; report n-gram overlap with every test set |
| Comparing on total params instead of active | All arms matched on **active** compute |
| Memory is decorative | Arm D — ablate memory, require degradation |
| Reinventing RAG at great expense | Arm E — RAG baseline over the same corpus, run **first** |
| Cherry-picked benchmarks | Pre-register the suite before running |
| "Skill" mistaken for intelligence | Chollet's control for priors/experience; Tier 4 and Tier 5 |
| Tuning on the test set across stages | Hold out a sealed set used **only** at stage gates |

**Contamination is the most likely way this project produces a false positive.** A memory bank written
from a large web corpus will contain benchmark items verbatim, and a system designed to retrieve
exact n-gram matches is *maximally* able to exploit that. N-gram overlap reporting is mandatory, not
optional.

---

## 8. The decision criterion

The system is worth continuing if and only if, at matched active compute:

1. It beats **Arm A** (dense) on Tier 1 — *and* beats **Arm E** (RAG) meaningfully; and
2. It does not **lose** on Tier 2 (reasoning must not be sacrificed for knowledge); and
3. **Arm D** confirms the memory is load-bearing; and
4. Accuracy per active parameter improves monotonically from S1 → S2 → S3.

Criterion 4 *is* the U-curve measurement from [09 §6](09_method_comparison_and_decision.md). If it
turns over before S3, the trillion-parameter target is abandoned on evidence — which is the correct
outcome, not a failure of the project.
