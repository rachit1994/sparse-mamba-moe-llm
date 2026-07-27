# 10 — The Dynamic Substrate: Why This Is Now the Primary Architecture

> Numbers from `verify_capacity.py`; output at `verify_capacity_output.txt`.
> ```
> $ python3 verify_capacity.py   → exit 0
> ```
> **This document reverses the architectural decision in [09](09_method_comparison_and_decision.md).**
> Memory-bank-first is demoted to a fallback. Dynamics-first is the primary path.

---

## 0. What changed and why

Docs 01–09 optimised a premise I had quietly replaced. The original proposal claims:

> Intelligence emerges from dynamic **patterns** formed over a **fixed** computational substrate.

I converted that into "store knowledge in an external table and retrieve it," which is a different
claim — storage and lookup, not dynamics. That substitution was wrong on two counts, one conceptual
and one quantitative.

**Conceptually:** retrieval fetches a *static value*. The original proposal's Component 5 (Pattern
Interaction) requires patterns to *interact and produce new patterns*. A lookup table cannot do that
by construction. I marked Component 5 "needs a falsifiable metric before it is buildable" and
Component 7 "deferred," then built the parts that were already solved elsewhere. That is not
optimisation; it is substitution.

**Quantitatively:** the memory bank was sized for a capacity problem that mostly does not exist.

---

## 1. The number that reversed the decision

Allen-Zhu & Li (*Physics of Language Models Part 3.3*, ICLR 2025) measured how much factual knowledge
a parameter can hold: **2 bits per parameter**, robust down to int8. Their own calibration: a **7B
model stores 14 Gbit — exceeding English Wikipedia and textbooks combined.**

Applied to this machine's 6.5 GB weight budget:

| Target | Bits | Fits in 52 Gbit? |
|---|---:|---|
| English Wikipedia + textbooks | 14.0 Gb | **YES — 3.7× headroom** |
| 30B-token web corpus (long tail) | 240.0 Gb | short by 4.6× |
| 300B-token corpus (the Pile) | 2400.0 Gb | short by 46.2× |

**A fixed brain resident entirely in RAM holds Wikipedia-scale knowledge with 3.7× headroom.**

And the tier comparison is damning for the old design:

| Tier | Bandwidth | Relative |
|---|---:|---:|
| Fixed brain (RAM) | 84 GB/s | **28×** |
| External bank (SSD) | 3 GB/s | 1× |

**The earlier architecture put 97% of parameters on the tier that is 28× slower.** That is the wrong
place for the load-bearing component on a bandwidth-bound machine. The premise, taken literally,
places knowledge in RAM — which is both faster and, per the capacity math, sufficient.

---

## 2. The unit error in the original capacity argument

The root README's capacity argument is wrong, but not in the way I first said. My earlier correction
([02](02_math_corrections.md)) fixed the arithmetic (10^189 → 10^241.8) and missed the deeper problem:
**the unit.**

```
C(10000,100)        = 10^241.8   ← a COUNT of distinguishable patterns
log2(C(10000,100))  = 803 bits   ← what one pattern can actually CARRY
```

A pattern count is an **address space**. Its logarithm is the storage capacity. 10²⁴¹ patterns is not
10²⁴¹ bits of knowledge — it is 803 bits. The hard ceiling for any n-unit substrate is n bits,
regardless of encoding:

| Units n | Active w | log₂ C(n,w) | Raw ceiling |
|---:|---:|---:|---:|
| 10,000 | 100 | 803 bits | 10,000 bits |
| 100,000 | 1,000 | 8,073 bits | 100,000 bits |
| 1,000,000 | 10,000 | 80,785 bits | 1,000,000 bits |

The SDR argument correctly establishes that **addressing** is not the bottleneck. That was never in
dispute. It says nothing about how many bits can be stored — which is the quantity that matters and
which is bounded by the substrate size.

**This does not sink the premise.** §1 shows the bit-bounded capacity is still enough for
Wikipedia-scale. The argument just needs to be stated in bits.

---

## 3. The lineage the proposal was pointing at

Three lines of work implement "fixed substrate, state carries the intelligence." All have code.

### 3.1 BDH — The Dragon Hatchling (Pathway, Sept 2025) — **MIT licensed**

*[arXiv:2509.26507](https://arxiv.org/abs/2509.26507) · [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)*

The closest published thing to the original proposal's Components 2 and 5.

- **Working memory during inference relies entirely on synaptic plasticity with Hebbian learning.**
  Not a lookup table — a dynamical state.
- **Attention *emerges*** from pairwise synapse updates governed by local correlation: neurons that
  fire together strengthen their synapse; opposing activation weakens it. Global attention-like
  behaviour arises from purely local rules rather than being imposed.
- Rivals GPT-2-architecture transformers at matched parameters, **10M → 1B**, same training data.
- **Monosemantic synapses and emergent modularity** — interpretability falls out of the architecture
  instead of requiring a post-hoc sparse autoencoder.
- Scale-free network structure; a GPU-friendly state-space formulation (BDH-GPU).
- Repo is small: `bdh.py`, `train.py`, `requirements.txt`, toy dataset. **MIT.**

**Caveat, stated plainly:** the open-source release is the paper's baseline variant. The headline
97.4% Sudoku-Extreme result comes from Pathway's *internal* implementation and **is not reproducible
from the public repo.** Do not plan against that number.

### 3.2 TTT — Learning to (Learn at Test Time) (Sun et al., ICML 2025)

*[arXiv:2407.04620](https://arxiv.org/abs/2407.04620) · PyTorch and JAX implementations released*

**The hidden state *is* a model**, and the update rule is a step of self-supervised learning. TTT-Linear
(state = linear model) and TTT-MLP (state = 2-layer MLP), evaluated 125M → 1.3B.

The decisive result: **TTT keeps reducing perplexity as context grows, where Mamba stops improving
after 16k.** A fixed-size state that *learns* beats a fixed-size state that merely *accumulates*.
This is the strongest evidence that "the state is the intelligence" is a real mechanism and not a
metaphor.

### 3.3 Modern Hopfield (Ramsauer et al., 2020)

Already cited in the proposal, with its significance understated: **attention *is* modern Hopfield
retrieval.** Attractor dynamics with exponential storage capacity is not an alternative to
transformers — it is what transformers already are. Component 2 is therefore proven at scale; it just
goes by a different name. The novel part is not attractor retrieval, it is attractor *interaction*.

### 3.4 Titans (Google, NeurIPS 2025)

Neural long-term memory that memorises at test time, using the **gradient of the loss as a surprise
signal**, with momentum and gated forgetting. Scales past 2M tokens. Already adopted for the memory
lifecycle in [04](04_architecture.md); here it is re-classified as *part of the dynamic core* rather
than an accessory to a retrieval store.

---

## 4. What this lineage has NOT shown — the honest gap

Being precise, because this is where the research risk actually sits:

**In every one of these systems, durable world knowledge still lives in trained weights.**

- BDH's Hebbian state is **working** memory — within-context. Its world knowledge is in its weights.
- TTT's and Titans' measured wins are **long-context** memorisation, not MMLU. The state excels at
  remembering *this conversation*, not *the world*.
- No published system stores durable facts in activation patterns rather than parameters.

So the defensible split is:

| Function | Where it demonstrably lives | Confidence |
|---|---|---|
| Reasoning, working memory, in-context adaptation | **Dynamics** (TTT, Titans, BDH) | High — published, reproducible |
| Durable world knowledge | **Weights** (in every published system) | High |
| **New concepts forming from pattern interaction** | **Nobody has shown this** | **This is the actual frontier** |

The third row is the original proposal's Component 5, and it is the genuine research contribution.
It is also the only part with zero published implementation — which is precisely why it is worth
doing and precisely why I previously avoided it.

---

## 5. The revised architecture: dynamics-first

**Primary: a fixed dynamic substrate, entirely in RAM.**

```
┌──────────────────────────────────────────────┐
│  FIXED SUBSTRATE  —  6.5 GB, RAM, 84 GB/s    │
│                                              │
│  · dense core (BDH-style Hebbian / TTT state)│
│  · world knowledge in weights (≤52 Gbit)     │
│  · working memory = synaptic/fast-weight     │
│  · pattern interaction happens HERE          │
└──────────────────────────────────────────────┘
                     │
                     │  only for what exceeds 52 Gbit
                     ▼
┌──────────────────────────────────────────────┐
│  OVERFLOW STORE (fallback)  —  SSD, 3 GB/s   │
│  · Engram n-gram hash, prefetched            │
│  · sized by MEASUREMENT, not by "1 trillion" │
└──────────────────────────────────────────────┘
```

The external store is now **conditional**, not central. It exists only if the measured knowledge
requirement exceeds the substrate's information ceiling. Everything in
[09](09_method_comparison_and_decision.md) about hash addressing and prefetch remains correct — it
is simply demoted from the main architecture to the overflow tier.

**Consequences:**

| | Old (memory-first) | New (dynamics-first) |
|---|---|---|
| Knowledge tier | SSD, 3 GB/s | **RAM, 84 GB/s** |
| Disk required | 200 GB | **~5–20 GB** |
| Machine needed | 512 GB / external NVMe | **current 256 GB machine** |
| Time to first signal | 3–6 months | **~2 weeks** |
| Tests the actual premise? | **No** | **Yes** |

---

## 6. The experiment that decides everything

**Measure knowledge bits per parameter on a dynamic substrate, against Allen-Zhu's 2.0 bits/param
dense baseline.** Nobody has published this number.

| Measured | Verdict |
|---:|---|
| < 1.0 b/param | **KILL** — dynamics is worse than plain weights for knowledge |
| 1.0–2.0 | **KILL for knowledge**; keep dynamics for reasoning only, add overflow store |
| = 2.0 | Parity — premise neither proven nor refuted; decide on other grounds |
| > 2.0 | **PREMISE PROVEN.** Publishable, and the whole programme is justified |

What a 6.5 GB fixed brain would then hold:

| Rate | Capacity | vs Wikipedia+textbooks |
|---:|---:|---:|
| 1.0 b/param | 32.5 Gbit | 2.3× |
| 2.0 b/param | 52.0 Gbit | 3.7× |
| 4.0 b/param | 52.0 Gbit (storage-bound) | 3.7× |

Note the ceiling: above ~1.6 bits/param the **physical storage bound** binds before the capacity law
does, at ternary precision. Beating 2.0 bits/param therefore requires higher precision, not just a
better architecture — a subtlety that must be controlled for in the experiment.

**Cost of this experiment: ~2 weeks, BDH at 10M–100M params, MIT-licensed code, a few dollars of
electricity, on the machine already owned.** Compare with 3–6 months for the memory-bank path.

---

## 7. Open risk

1. **Allen-Zhu's 2 bits/param is unverified below int8.** Ternary rows in §1 are upper bounds, not
   measurements. Gate **G-CAP** must measure it directly before any ternary capacity claim is made.
2. **BDH's public repo is the baseline variant.** Its headline Sudoku result is not reproducible from
   released code. Plan against the paper's language-modelling numbers only.
3. **Hebbian/attractor training is historically unstable at scale.** The field moved Hopfield →
   attention for a reason. BDH claims to have solved this at ≤1B; that claim is exactly what
   reproduction must check.
4. **Component 5 remains unimplemented by anyone.** Everything above makes it *testable*; none of it
   makes it *solved*.

---

## Sources

- [Physics of Language Models Part 3.3: Knowledge Capacity Scaling Laws, arXiv:2404.05405](https://arxiv.org/abs/2404.05405)
- [The Dragon Hatchling, arXiv:2509.26507](https://arxiv.org/abs/2509.26507) · [code, MIT](https://github.com/pathwaycom/bdh)
- [Learning to (Learn at Test Time), arXiv:2407.04620](https://arxiv.org/abs/2407.04620) · [PyTorch](https://github.com/test-time-training/ttt-lm-pytorch)
- [Hopfield Networks is All You Need, arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- [Titans: Learning to Memorize at Test Time, arXiv:2501.00663](https://arxiv.org/pdf/2501.00663)
