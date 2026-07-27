# 04 — Architecture Specification

> **Revised** after the prior-art research in [08](08_prior_attempts_and_code.md) and the method
> comparison in [09](09_method_comparison_and_decision.md). Where earlier drafts of this file
> conflict, [09](09_method_comparison_and_decision.md) wins.
>
> Numbers from `verify_math.py` §8–§9 and `verify_decision.py` §6.

---

## 0. What changed from the first draft

The first version of this spec was written before the prior-art survey and contained four concrete
errors. They are recorded rather than silently fixed:

| First draft | Corrected | Source of correction |
|---|---|---|
| 8 memory layers | **~3 access points** | Meta: degrades past ~3 |
| Separate memory pool per layer | **One shared pool** | Meta: shared pool, fixed param count |
| Learned product keys as primary addressing | **Deterministic n-gram hash** | Prefetchability ([09 §3](09_method_comparison_and_decision.md)) |
| top-k=128 scattered gather | **Hash lookup + block fetch, two tiers** | ~85× cheaper amortised |
| 1T parameters as the spec | **Staged; 1T gated on measurement** | Three diminishing-returns results |

---

## 1. Design constraints, restated

From [01](01_feasibility.md), the constraints that actually bind:

1. **RAM budget: 9.66 GB.** Anything not in it must be on SSD.
2. **Decode is bandwidth-bound**, not compute-bound. Active bytes/token is the throughput dial.
3. **SSD is 3 GB/s sequential; random IOPS is unmeasured.** Do not put an unmeasured quantity on the
   critical path.
4. **Prefill dominates user-visible latency** (4.84 s @ 2048 tokens), so anything that batches
   prefill IO is worth more than an equivalent decode gain.

Design rule that follows: **every SSD access must have an address that is knowable before the layer
that needs it executes.**

---

## 2. The build target: Stage S2 (126B parameters)

S2 is the primary build. S1 is a warm-up; S3/S4 are gated on measurement ([09 §6](09_method_comparison_and_decision.md)).

| Component | Params | Footprint | Tier | Notes |
|---|---:|---:|---|---|
| Dense backbone (SSM + sparse attention) | 2.0 B | 0.40 GB | RAM | the reasoning anchor — **do not shrink** |
| Fine-grained MoE pool | 24.0 B | 4.80 GB | RAM | **fully resident** ⇒ zero expert SSD traffic |
| — of which active per token | 0.6 B | 0.12 GB | — | |
| Tier-A memory: n-gram hash bank | ~70 B | 14.0 GB | SSD | prefetched, deterministic |
| Tier-B memory: hierarchical blocks | ~30 B | 6.0 GB | SSD | one block per context |
| **Total** | **~126 B** | **25.2 GB disk** | | fits the base 256 GB machine |

RAM ledger, verified:

```
backbone + experts resident        5.20 GB
KV cache (32k ctx, 4 attn layers)  0.54 GB
activations / scratch              0.50 GB
-----------------------------------------
total                              6.24 GB   of 9.66 GB budget
slack                              3.42 GB
```

Throughput, verified:

```
[bandwidth] 0.520 GB/token / 84 GB/s  = 6.19 ms → 161.6 tok/s   ← binding
[compute  ] 5.20 GFLOP/token          = 3.94 ms → 253.8 tok/s
[memory IO] prefetched                → hidden under compute
plan against 50% of serialized                  →  ~37 tok/s
```

**The whole expert pool is resident.** This is the single most important structural choice: it
converts a fragile cache-hit-rate dependency (which is what makes Kimi K2 unusable at 0.47–4.7 tok/s)
into a hard guarantee. Only the memory tiers touch SSD, and both are prefetchable.

---

## 3. Component specification

### 3.1 Dense backbone — 2.0 B, RAM-resident

Mamba-2 / SSM layers with a small number of interleaved attention layers.

- **SSM for O(1) state** — no KV-cache growth with sequence length, which is what makes 32k+ context
  affordable in 0.54 GB.
- **4 attention layers** for global retrieval, in a Gemma-style local:global ratio (5:1 sliding-window
  to global). Gemma 4 uses 512-token windows on its E-series; adopt that as the starting point.
- **Shared KV cache** across the top layers (Gemma 4 reports 37.5% global KV reduction by reusing
  keys as values in global attention).
- Ternary weights (BitNet-style), **trained natively ternary** — not post-training quantized
  ([03 C1/C2](03_prior_art.md)).

**Do not shrink this to make room for memory.** Every surveyed result says the dense core is the
reasoning anchor and that dense/sparse are complementary ([09 §7](09_method_comparison_and_decision.md)).

### 3.2 Fine-grained MoE pool — 24 B, RAM-resident

- Fine granularity per Krajewski/Ludziejewski et al. — matching expert size to the FFN layer is
  **not optimal at almost any compute budget**.
- ~0.6 B active per token.
- **Entire pool resident at 4.80 GB**, so routing never causes an SSD read.

### 3.3 Tier-A memory: n-gram hash bank — ~70 B, SSD, prefetched

The primary knowledge store. Engram-style.

- **Addressing:** multi-head hash of suffix n-grams (N=2, N=3) over token IDs. **No learned keys,
  no index, no re-indexing.**
- **Injection:** residual branch into early-to-mid blocks — DeepSeek uses layers 2 and 15 of 36;
  scale proportionally. Injected *before* the attention block, not at the input.
- **Access points:** ~3 (Meta's measured sweet spot over a shared pool).
- **IO:** ~24 reads/token, **0.08 ms**, fully hidden under the 6.19 ms compute step.
- **Prefill:** all n-gram keys for the entire prompt are known before compute starts ⇒ issue every
  read as one batched, sorted, near-sequential pass.

Why this and not product keys: it is the only addressing scheme whose throughput is **flat at
161.6 tok/s across a 25× swing in SSD IOPS** ([09 §3](09_method_comparison_and_decision.md)).

### 3.4 Tier-B memory: hierarchical block memory — ~30 B, SSD, block-fetched

Semantic/long-tail knowledge, Apple-style.

- **Addressing:** hierarchical k-means over context. Apple's levels: (128 tok, 16 clusters),
  (32 tok, 32 clusters), (8 tok, 128 clusters).
- **Fetch:** one **contiguous block** per context — 225 M params ≈ 45 MB, one sequential read,
  15.0 ms per context switch = **0.015 ms/token** amortised over a 1000-token generation.
- Complements Tier A: hash lookup captures local lexical structure, block fetch captures topical
  knowledge. They fail in opposite directions, which is why both are present.

### 3.5 Memory lifecycle

Adopt Titans' **gradient-based surprise metric** (loss gradient as novelty signal, with momentum and
a gated forgetting mechanism) rather than the hand-specified `U = Confidence × Usage × Recency`
decay, which has no fitted constant and no measurement procedure ([02](02_math_corrections.md)).

### 3.6 Reasoning

Test-time compute over the throughput headroom. At ~37 tok/s, 10⁴ thinking tokens = **271 s per
answer**; at the 161 tok/s ceiling, 62 s. Energy is irrelevant (~1 Wh per 2-minute answer); wall-clock
is the only cost.

---

## 4. Data flow

```
tokens
  │
  ├─► n-gram hash (N=2,3, multi-head)  ──► PREFETCH Tier-A rows from SSD ─┐
  │   (addresses known here, before any compute)                          │
  │                                                                       │
  ├─► context k-means cluster ──► PREFETCH Tier-B block (once per context)┤
  │                                                                       │
  ▼                                                                       │
dense backbone (SSM + sparse attention, ternary, RAM)                     │
  │                                                                       │
  ├── block 2  ◄── Tier-A residual injection ◄──────────────────────────--┤
  ├── block k  ◄── Tier-B block add ◄────────────────────────────────────-┤
  ├── block 15 ◄── Tier-A residual injection ◄───────────────────────────-┘
  │
  ├─► fine-grained MoE routing (all experts resident in RAM)
  │
  ▼
output  ──► [optional] test-time search / reasoning loop
```

The two prefetch arrows are the architecture. Everything else is conventional.

---

## 5. Scaling path

| Stage | Memory | Total | Disk | Machine | Gate to proceed |
|---|---:|---:|---:|---|---|
| S1 | 30 B | 56 B | 11.2 GB | current | pipeline works end-to-end |
| **S2** | **100 B** | **126 B** | **25.2 GB** | **current** | **marginal-value curve still rising** |
| S3 | 300 B | 326 B | 65.2 GB | current | curve still rising at S2→S3 |
| S4 | 974 B | 1000 B | 200.0 GB | **512 GB / ext. NVMe** | curve has not turned by S3 |

S1–S3 all run on the machine already owned. **Only S4 requires a purchase, and S4 is gated on a
measurement not yet taken.**

---

## 6. What is unproven in this design

Stated plainly, because these are the things that make it research rather than engineering:

1. **The two-tier memory combination is novel.** Engram (hash) and Apple (blocks) have each been
   validated alone; nobody has published them together. They may be redundant rather than
   complementary.
2. **974B is 7.6× beyond the largest published memory bank.** S4 is an extrapolation, which is why
   it is gated.
3. **Native-ternary training at this scale on one machine is unvalidated.** BitNet's results are at
   2B dense.
4. **The dense/sparse balance is guessed.** 2B dense : 24B experts : 100B memory is not derived from
   a fitted scaling law — it is a starting point to be tuned against the U-curve.
5. **Pattern Synthesis (original Component 7) has no implementation.** No surveyed work does this at
   scale. It is deferred, not solved.
