# 08 — Who Has Actually Tried This, What Code Exists, What They Learned

> Research question: *has anyone built "knowledge in memory, not weights" at scale, did it work, and can I use their code?*
>
> **Answer: yes — five major labs, all within the last 24 months, and four released code.**
> The premise is no longer speculative. What is still unproven is the *ratio* this proposal wants.

---

## 0. The short version

| Lab | System | Memory params | Result | Code |
|---|---|---:|---|---|
| **Google** | Gemma 3n / Gemma 4 PLE | ~3.5B PLE | **Shipped in production**, twice | weights open |
| **DeepSeek** | Engram | 27B | Beats iso-param **and** iso-FLOP MoE | [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) |
| **Meta** | Memory Layers at Scale | 128B | Beats dense with **2× the compute** | [facebookresearch/memory](https://github.com/facebookresearch/memory) (**CC-BY-NC**) |
| **ByteDance** | UltraMem / UltraMemV2 | 120B | Matches 8-expert MoE, **2–6× faster** | not located |
| **Apple** | Hierarchical Memories | 4.6B bank (→21B) | 160M model ≈ 2× larger model | [apple/ml-memory-pretraining](https://github.com/apple/ml-memory-pretraining) |

**Nobody has exceeded ~128B memory parameters.** This proposal's 974B is an **8× extrapolation
beyond the largest published result**, and two of the five papers report evidence that returns
diminish before that point (§6).

---

## 1. Google — the premise is already shipping (Gemma 3n → Gemma 4)

This is the most important single fact in this document: **Google shipped this idea in a production
on-device model, then kept it for the next generation.**

**Per-Layer Embeddings (PLE).** A second embedding table feeds a small, low-dimensional residual
signal into *every* decoder layer. For each token, PLE combines a token-identity component (second
embedding lookup) with a context-aware component (learned projection of the main embeddings).

The critical engineering property, in Google's own framing:

> PLE data can be generated separately, **outside the operating memory of the model, cached to fast
> storage**, and then added to the model inference process as each layer runs.

That is precisely the architecture this proposal is reaching for — parameters that live on storage,
not in accelerator memory, fetched deterministically per layer.

**The numbers:**

| Model | Raw params | Effective params | Ratio | Notes |
|---|---:|---:|---:|---|
| Gemma 3n E2B | 5 B | 1.91 B | 2.6 : 1 | PLE caching + parameter skipping |
| Gemma 3n E4B | 8 B | ~4.5 B | 1.8 : 1 | |
| Gemma 4 E2B | 5 B | 2.3 B | 2.2 : 1 | PLE retained from Gemma 3 |
| Gemma 4 E4B | 8 B | 4.5 B | 1.8 : 1 | PLE retained |
| Gemma 4 26B-A4B | 26 B | 3.8 B activated | 6.8 : 1 | MoE variant |

PLE dimension is **256** against an FFN intermediate dimension of **16384** in gemma-3n-E4B-it —
the memory pathway is deliberately narrow and cheap.

Gemma 4 also ships **shared KV cache** (last N layers reuse earlier layers' KV; reusing keys as
values in global attention cuts the global KV cache **37.5%**), **dual RoPE** (1M frequency global /
10k local), and a **5:1 local-sliding to global attention ratio** (512-token window on E-series).

### What this calibrates

**Google, with effectively unlimited compute, shipped a raw:effective ratio of only 1.8–2.6 : 1.**
This proposal's design in [04](04_architecture.md) uses **37.5 : 1** (974B written : 26B trained).

One caveat against over-reading that gap: Google's PLE is a *production on-device* choice optimised
for tight, predictable RAM, not an attempt to find the sparsity frontier. The research frontier is
more aggressive — **UltraMemV2 runs at 48 : 1** (120B total / 2.5B activated), which is *more*
aggressive than this proposal's 37.5 : 1.

So the ratio is **not** the problem. Measured against published work:

| | This proposal | Most aggressive published | Verdict |
|---|---:|---:|---|
| Memory : active ratio | 37.5 : 1 | 48 : 1 (UltraMemV2) | **within range** |
| Absolute memory bank | 974 B | 128 B (Meta) | **7.6× beyond anything published** |

**The extrapolation is in absolute scale, not in sparsity.** That is a more tractable objection —
but it is compounded by the diminishing-returns evidence in §6, and
[09](09_method_comparison_and_decision.md) changes the target accordingly.

*Note: Gemma 4 shipped July 2026 (arXiv:2607.02770), after this project began; the PLE continuity
from 3n to 4 is the strongest available signal that the technique is durable rather than a one-off.*

---

## 2. DeepSeek — Engram (the most directly usable result)

*"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models",
DeepSeek-AI, [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) — code:
[github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)*

**Technique.** Modernized N-gram embedding with **O(1) lookup**. At each token position, suffix
n-grams (N=2, N=3) of recent tokens are hashed by a multi-head hash function into a fixed-size
embedding table. Injected as a **residual branch into early-to-mid blocks** (e.g. layers 2 and 15
of 36), immediately before the attention blocks — *not* at the input layer.

**Framing:** Conditional *Memory* as a second axis of sparsity, complementing MoE's conditional
*Computation*. Transformers "lack a native primitive for knowledge lookup, forcing them to
inefficiently simulate retrieval through computation."

**Results** — vs. a strictly iso-parameter **and** iso-FLOP MoE baseline, at 27B Engram params:

| Benchmark | Gain |
|---|---:|
| MMLU | +3.4 |
| CMMLU | +4.0 |
| BBH | +5.0 |
| ARC-Challenge | +3.7 |
| HumanEval | +3.0 |
| MATH | +2.4 |
| Multi-Query NIAH | 84.2 → **97.0** |

Two learnings that change this project's design:

1. **The gains are NOT confined to factual recall.** Reasoning (BBH +5.0), code (+3.0) and math
   (+2.4) improved *more* than knowledge (MMLU +3.4). This is direct counter-evidence to the
   "memory buys knowledge, not reasoning" risk stated in [01](01_feasibility.md) and
   [03](03_prior_art.md). The mechanism they propose: delegating local dependencies to lookup
   **frees attention capacity for global context** — hence the NIAH jump. **That risk must be
   downgraded, and my earlier statement of it was too strong.**
2. **Deterministic addressing enables runtime prefetching from host memory at negligible overhead.**
   This is the single most important engineering sentence found in this entire research effort —
   see [09 §3](09_method_comparison_and_decision.md).

**Also reported: a U-shaped scaling law for "Sparsity Allocation"** — the optimal trade-off between
neural computation (MoE) and static memory (Engram). A U-curve means **there is an optimum and
returns turn negative past it.** See §6.

---

## 3. Apple — Hierarchical Memories (closest to our exact situation)

*Pouransari, Grangier, Thomas, Kirchhof, Tuzel (Apple), ICLR 2026,
[arXiv:2510.02375](https://arxiv.org/abs/2510.02375) — code:
[github.com/apple/ml-memory-pretraining](https://github.com/apple/ml-memory-pretraining)*

Same company as the target hardware, and **explicitly motivated by edge devices with limited
inference-time memory and compute.** Their framing of the problem is nearly identical to this
proposal's:

> compressing all world knowledge into parameters is unnecessary, as only a fraction is used per
> prompt, and impractical for edge devices

**Technique.** A small LM accesses a **large hierarchical parametric memory bank**. Per context, it
fetches **one small memory block** and *adds* it to the model. Pretraining pushes long-tail world
knowledge into memory parameters while the small LM acts as an **anchor for common knowledge and
general reasoning**.

Hierarchy is built by k-means clustering over context:

| Level | Tokens | Clusters |
|---|---:|---:|
| 1 | 128 | 16 |
| 2 | 32 | 32 |
| 3 | 8 | 128 |

**Headline result:** a **160M-param model + 18M-param fetched memory, drawn from a 4.6B memory bank**,
matches models with **more than 2× the parameters**, at trillion-token scale. Memories scaled to
**over 21B parameters**.

**The design lesson that matters most here:** Apple fetches a **contiguous block**, not top-k
scattered vectors. On SSD-backed storage that is the difference between one sequential read and a
thousand random reads. See [09 §4](09_method_comparison_and_decision.md).

Repo provides `train_kmeans.py`, `train_memory.py`, `eval_memory.py`, config examples, and requires
an llmfoundry patch for evaluation.

---

## 4. ByteDance — UltraMem / UltraMemV2 (the memory-access argument)

*UltraMem: "Ultra-Sparse Memory Network", ICLR 2025, [arXiv:2411.12364](https://arxiv.org/abs/2411.12364).
UltraMemV2: [arXiv:2508.18756](https://arxiv.org/abs/2508.18756)*

This line exists for exactly the reason this project cares about: **MoE has high memory-access cost
at inference; memory layers do not.** On a 120 GB/s bandwidth-bound box, that is the whole game.

**UltraMem (v1):**
- **2–6× faster inference than MoE**, up to **83% lower inference cost**
- Inference time stays flat as total parameters grow, provided activated params are constant —
  where MoE degrades sharply
- A model with **12× the parameters of a 1.6B dense model matches a 6.5B dense model** while keeping
  1.6B-dense compute
- Trained at **20 million values**
- **TDQKR** (Tucker-Decomposed Query-Key Retrieval) replaces product quantization for more precise
  index recall, at negligible extra parameters
- **IVE** (Implicit Value Expansion, E=4) virtually expands the memory table, cutting memory access
- Memory layers are **split into segments distributed across the transformer**, so transformer
  compute and memory fetch **overlap** — latency hiding by construction

**UltraMemV2:** five changes — memory layers in **every** transformer block, single-linear-projection
value expansion, **FFN-based value processing borrowed from PEER**, principled initialization, and
rebalanced memory-to-FFN compute ratio. Reaches **parity with 8-expert MoE** at equal compute and
params with far lower memory access (v1 only matched 2-expert MoE). Scales to **2.5B activated /
120B total** (48:1). Gains: long-context memorization **+1.6**, multi-round memorization **+6.2**,
in-context learning **+7.9**.

**No code located.** Techniques are reimplementable from the papers; budget for that.

---

## 5. Meta — Memory Layers at Scale (code released, license-restricted)

*[arXiv:2412.09764](https://arxiv.org/abs/2412.09764), ICML 2025 — code:
[github.com/facebookresearch/memory](https://github.com/facebookresearch/memory), built on Meta Lingua*

Covered in [03](03_prior_art.md). Three implementation facts that only surfaced from the repo and
follow-ups, and that **contradict the architecture I specified in [04](04_architecture.md)**:

1. **The memory pool is SHARED across all memory layers** — same parameters, multiple access points.
   This maximises parameter sharing and keeps the count fixed. I had assumed separate pools.
2. **~3 memory layers is the sweet spot.** Performance rises with more memory layers up to about 3,
   then **degrades as further FFN layers are replaced** — "sparse and dense layers are both needed
   and likely complementary." **I specified 8. That was wrong on the evidence available.**
3. Reference config is `pkplus_373m_1024k.yaml` — a **373M** base model with **1024k = 2²⁰ memory
   slots**. Their demonstrated design point is far smaller than what this proposal assumes.

**License: CC-BY-NC — non-commercial.** Usable for research; **cannot** ship in a commercial product.
This is a hard constraint, not a footnote. DeepSeek's and Apple's repos should be checked before
adopting either as the base.

*Note the tension with UltraMemV2, which puts memory in **every** block. UltraMemV2 is newer and
explicitly redesigns value processing and initialization to make dense placement work — see
[09 §5](09_method_comparison_and_decision.md) for how this is resolved.*

---

## 6. What went wrong historically — and why it is fixed now

**Product-key memory (Lample et al. 2019) worked and was then abandoned.** The original result was
strong: a memory-augmented **12-layer** model outperformed a **24-layer** baseline while being **2×
faster at inference**. It did not get adopted for a decade-defining reason:

> it's a challenge to incorporate [fast approximate vector similarity] when the keys are being
> **continually trained and need to be re-indexed**

That is the failure mode. **Learned keys and fast indexes are mutually hostile** — every gradient
step invalidates the index.

**This is exactly what Engram's hash addressing eliminates.** N-gram hashes are computed from token
IDs. They are never trained, never re-indexed, and are known *before the model runs*. Ten years of
stalled adoption traces to one design decision that DeepSeek reversed.

**The second historical failure — scale claims outrunning demonstrations.** PEER is titled *Mixture
of a Million Experts*, but its own limitations note that scalability to a literal million experts
"may be challenging in practice" and the paper "does not provide a concrete demonstration of this
scale." No reproduction failures were located, but **no independent confirmation at that scale was
located either.** Treat PEER's headline number as an architecture proposal, not a validated result.

**The third signal — two independent diminishing-returns findings.** These are the strongest
evidence against this proposal's 974B target:

| Source | Finding |
|---|---|
| UltraMemV2 | **UltraMemV2-2.5B/60B-top768 outperforms UltraMemV2-2.5B/120B-top256.** At fixed activated compute, *tripling activation density beats doubling total sparse parameters.* "Activation density has greater impact on performance than total sparse parameter count." |
| DeepSeek Engram | **U-shaped scaling law** for sparsity allocation — an optimum exists, and allocating past it is net negative. |
| Meta Memory Layers | Performance degrades past ~3 memory layers as dense capacity is removed. |

**Read together: total sparse parameter count is the *least* valuable axis to scale.** The proposal's
headline metric — one trillion parameters — is close to the *worst* thing to optimise for. This does
not refute the premise; it refutes the target. See [09](09_method_comparison_and_decision.md).

---

## 7. Code inventory — what can actually be used

| Repo | What it gives | License | Verdict |
|---|---|---|---|
| [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) | n-gram hash conditional memory, 27B-scale validated | **verify before use** | **Primary reference** |
| [apple/ml-memory-pretraining](https://github.com/apple/ml-memory-pretraining) | hierarchical k-means memory, block fetch, train+eval scripts | **verify before use** | **Primary reference** |
| [facebookresearch/memory](https://github.com/facebookresearch/memory) | product-key memory on Meta Lingua, `pkplus_373m_1024k.yaml` | **CC-BY-NC** | research only — blocks commercial use |
| [lucidrains/product-key-memory](https://github.com/lucidrains/product-key-memory) | standalone `PKM` module (`dim, heads, num_keys, topk`) | permissive | good for ablations |
| [lucidrains/PEER-pytorch](https://github.com/lucidrains/PEER-pytorch) | `PEER` block (`num_experts, num_experts_per_head, dim_key`) | permissive | good for ablations |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | Titans test-time memory (unofficial) | permissive | for the lifecycle component |
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | `bitnet.cpp` ternary inference kernels | check | ternary path |
| Gemma 3n / Gemma 4 weights | PLE in a shipped model, inspectable | Gemma terms | **reference implementation to study** |
| MLX / `mlx-lm` | unified-memory arrays, 4-bit quant, LoRA/QLoRA, lazy eval | MIT | **the Mac runtime** |

**Licensing action required before Phase 1:** confirm the actual license on the DeepSeek and Apple
repos. If both are restrictive, the fallback is a clean-room implementation from the papers —
Engram's hash-lookup mechanism is simple enough that this is genuinely tractable, which is a further
argument for choosing it.

---

## 8. What this changes

1. **The premise is validated.** Five labs, four codebases, consistent direction. Stop arguing for it
   and start measuring it.
2. **"Memory buys knowledge, not reasoning" was too strong** — Engram's reasoning gains exceeded its
   knowledge gains. Correcting that claim from [01](01_feasibility.md)/[03](03_prior_art.md).
3. **Learned product keys are the wrong primitive** for SSD-backed inference. Deterministic hash
   addressing is prefetchable; learned keys are not.
4. **Block fetch beats top-k gather** on this hardware.
5. **The 1T target is the weakest part of the proposal** and should be replaced by a
   measured optimum. The premise survives; the headline number should not.
6. **My [04](04_architecture.md) spec has concrete errors** — 8 memory layers (should be ~3, or every
   block under UltraMemV2's redesign), unshared pools (should be shared), learned keys (should be
   hashed), top-k gather (should be block fetch).

Corrections and the resulting decision: [09_method_comparison_and_decision.md](09_method_comparison_and_decision.md).
