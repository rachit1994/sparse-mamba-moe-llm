# 04 — Architecture Specification (Dynamics-First)

> **Rewritten.** The previous version of this file specified a memory-bank-first architecture with
> 974B parameters on SSD. [10_dynamic_substrate.md](10_dynamic_substrate.md) reversed that decision on
> capacity and bandwidth grounds. This file now specifies the dynamics-first architecture; the memory
> bank survives as a conditional overflow tier.
>
> Numbers from `verify_capacity.py` and `verify_math.py`.

---

## 0. Revision history of this spec

| Version | Design | Why it changed |
|---|---|---|
| v1 | Product-key memory, 8 layers, 974B on SSD | 8 layers wrong (Meta: ~3), pools unshared, keys not prefetchable |
| v2 | Engram hash + block fetch, 974B on SSD | Correct mechanism, **wrong tier** — put 97% of params on the 28×-slower device |
| **v3 (this)** | **Fixed dynamic substrate in RAM; overflow store conditional** | Capacity math: 6.5 GB RAM holds Wikipedia+textbooks with 3.7× headroom |

Keeping this table because the errors are instructive: v1 was wrong about mechanism, v2 was wrong
about placement. Both were confidently specified.

---

## 1. The constraint that determines the design

From [01](01_feasibility.md) and [10](10_dynamic_substrate.md):

| | Capacity | Bandwidth |
|---|---:|---:|
| RAM (6.5 GB weights) | 52 Gbit | 84 GB/s |
| SSD | ~1600 Gbit | 3 GB/s |
| **Wikipedia + textbooks** | **14 Gbit** | — |

**Design rule: knowledge goes in RAM until 52 Gbit is exhausted. Only genuine overflow goes to SSD.**

The v2 design violated this by putting 194 GB of knowledge behind a 3 GB/s interface when 14 Gbit of
it would have fit in an 84 GB/s interface.

---

## 2. The build target

```
┌────────────────────────────────────────────────────────────────┐
│  FIXED SUBSTRATE — all resident, 6.5 GB, 84 GB/s               │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Dense core          ~2–4 B params      0.4–0.8 GB        │  │
│  │   BDH-style Hebbian synaptic state, or TTT-style          │  │
│  │   learned state. Carries reasoning + working memory.      │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Knowledge weights   ~26 B params (ternary)  5.2 GB        │  │
│  │   ≤52 Gbit of durable world knowledge.                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Fast-weight / synaptic state    ~0.2 GB                   │  │
│  │   Written at inference. Hebbian or TTT update rule.       │  │
│  │   THIS is where pattern interaction happens.              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
                              │  ONLY if measured knowledge need > 52 Gbit
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  OVERFLOW STORE — SSD, conditional, sized by measurement       │
│  Engram n-gram hash, deterministic addressing, prefetched.     │
│  Mechanism per [09]; tier demoted from primary to fallback.    │
└────────────────────────────────────────────────────────────────┘
```

RAM ledger:

```
dense core + knowledge weights     5.60–6.00 GB
fast-weight / synaptic state       0.20 GB
KV cache (or none, if pure SSM)    0.00–0.54 GB
activations / scratch              0.50 GB
------------------------------------------------
total                              6.30–7.24 GB   of 9.66 GB budget
```

Fits, with 2.4–3.4 GB slack. **Disk required: ~5–20 GB, not 200 GB. The current 256 GB machine is
sufficient — no hardware purchase.**

---

## 3. Components

### 3.1 Dense core — the dynamic substrate

Two candidate mechanisms; **Phase 1 measures both and picks one.** Do not pre-commit.

**Option A — BDH (Hebbian synaptic state).** Working memory lives entirely in synaptic plasticity
with local Hebbian updates; attention-like behaviour *emerges* from pairwise correlation rather than
being imposed. Gives monosemantic synapses and emergent modularity for free, which makes
[06](06_evaluation.md)'s interpretability metrics measurable without a separate SAE.
MIT-licensed, `bdh.py` + `train.py`, validated 10M–1B.

**Option B — TTT (state as a learned model).** The hidden state *is* a model; the update rule is a
self-supervised gradient step. Strongest published evidence that a fixed state can keep absorbing
information: perplexity keeps falling past 16k where Mamba plateaus. PyTorch and JAX releases.

**Selection criterion:** bits-of-knowledge per parameter (§5), then tok/s at matched quality. Not
aesthetics.

### 3.2 Knowledge weights

Ordinary trained parameters holding durable world knowledge, ternary where G-CAP permits. Capped at
52 Gbit by physics.

**Open issue — G-CAP:** Allen-Zhu's 2 bits/param is verified only down to int8. At ternary the
storage bound (1.6 bits/param) binds *before* the capacity law, so **beating 2.0 bits/param requires
int8 or higher, not just a better architecture.** Any capacity claim at ternary must be measured, not
assumed. This must be controlled in the Phase-1 experiment or the result is uninterpretable.

### 3.3 Fast-weight / synaptic state — where the premise lives

The component the original proposal is actually about, and the one v1/v2 omitted.

- Updated **at inference**, no gradient step on the base weights.
- Hebbian (BDH) or self-supervised (TTT) or surprise-gated (Titans).
- **Pattern interaction — original Component 5 — happens here or nowhere.** Two active patterns
  co-occurring modify shared synapses; the modification *is* the new pattern.
- Forgetting: Titans' gradient-of-loss surprise signal with momentum and gating. Not a hand-tuned
  decay constant ([02](02_math_corrections.md)).

### 3.4 Overflow store — conditional

Everything in [09](09_method_comparison_and_decision.md) stands: deterministic n-gram hash
addressing, prefetched, ~24 reads/token, flat throughput from 500K down to 20K IOPS.

**It is only built if Phase 2 measures a knowledge requirement above 52 Gbit.** Building it before
that measurement is what v2 did wrong.

---

## 4. What is unproven

1. **No published system stores durable world knowledge in dynamic state.** BDH/TTT state is
   *working* memory. This spec assumes the dense weights carry world knowledge and the dynamic state
   carries reasoning and working memory — which is the published, conservative reading.
2. **Pattern interaction (Component 5) has no implementation anywhere.** §3.3 says where it would
   live, not that it works.
3. **BDH's public repo is the paper baseline**; the 97.4% Sudoku figure is from an unreleased
   internal implementation and must not be planned against.
4. **Hebbian training stability above 1B is unvalidated.**
5. **Ternary + capacity law interaction (G-CAP)** could invalidate any bits/param comparison.

---

## 5. The decisive measurement

**Knowledge bits per parameter, dynamic substrate vs dense baseline (2.0 b/param, Allen-Zhu).**

| Result | Action |
|---:|---|
| < 1.0 | Kill dynamics-for-knowledge; revert to v2 memory-bank design |
| 1.0–2.0 | Dynamics for reasoning only; build overflow store |
| = 2.0 | Parity; decide on throughput and interpretability |
| > 2.0 | **Premise proven.** Publishable; scale the substrate |

Cost: ~2 weeks at 10M–100M params on the current machine. This replaces the 3–6 month memory-bank
programme as the first thing built.
