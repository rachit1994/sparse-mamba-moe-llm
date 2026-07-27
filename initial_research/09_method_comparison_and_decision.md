# 09 — Method Comparison and Decision

> Numbers from `verify_decision.py`; output committed at `verify_decision_output.txt`.
> ```
> $ python3 verify_decision.py   → exit 0
> ```
> Decisions here **supersede** the architecture in [04](04_architecture.md) where they conflict.

---

## 0. The decision, up front

| Question | Decision |
|---|---|
| **Memory addressing** | **Deterministic n-gram hash (Engram-style)** as the primary path — *not* learned product keys |
| **Storage layout** | **Contiguous block fetch (Apple-style)** for the semantic tier — *not* top-k scattered gather |
| **Memory layer placement** | **Shared pool, ~3 access points**, revisit toward every-block only after UltraMemV2's redesign is reproduced |
| **Parameter target** | **Staged: 126B → 326B → 1T**, with continuation *gated on a measured U-curve* — 1T becomes an outcome, not a spec |
| **Base codebase** | **DeepSeek Engram + Apple ml-memory-pretraining** as references; Meta's is CC-BY-NC and cannot ship commercially |
| **What is preserved** | The premise: *knowledge lives in memory, weights are infrastructure, intelligence comes from patterns over a fixed substrate* |

**The premise survives intact and is now well-supported. The technique and the headline number change.**

---

## 1. The candidate methods

Seven mechanisms can implement "parameters that live outside the fast path." They are not
interchangeable — they differ on the one axis this hardware cares about: **whether the address is
knowable before the layer runs.**

| Method | Address derived from | Lookup cost | Keys trained? | Prefetchable? | Max published | Code |
|---|---|---|---|---|---:|---|
| **PKM** (Lample 2019) | residual stream | O(√N) | yes | **no** | 2²⁰ slots | lucidrains |
| **PEER** (DeepMind 2024) | residual stream | O(√N) | yes | **no** | 1M experts *(claimed, not demonstrated)* | lucidrains |
| **Memory Layers** (Meta 2024) | residual stream | O(√N) | yes | **no** | **128 B** | facebookresearch/memory (**CC-BY-NC**) |
| **UltraMem/V2** (ByteDance 2025) | residual stream, TDQKR | O(√N) | yes | partial | **120 B** | not located |
| **Engram** (DeepSeek 2026) | **token IDs (n-gram hash)** | **O(1)** | **no** | **YES** | 27 B | deepseek-ai/Engram |
| **Hierarchical memories** (Apple 2026) | **context cluster (k-means)** | O(clusters) | centroids only | **YES (block)** | 21 B | apple/ml-memory-pretraining |
| **PLE** (Google, shipped) | **token ID** | **O(1)** | table only | **YES** | ~3.5 B | Gemma weights |

Note the pattern in the right-hand columns: **every method that is prefetchable derives its address
from something known before inference (token IDs, or a context cluster), and every method that is not
derives it from the residual stream.** That is not a coincidence; it is the whole decision.

---

## 2. Why the residual-stream methods lose *on this specific machine*

This is not a claim that product-key memory is bad. Meta got 128B parameters working with it, which
is more than anyone else. It is a claim about a **hardware-specific** constraint.

A learned-key memory computes its query from the residual stream at layer *L*. The address therefore
cannot be known until layers 1…L−1 have executed. **The SSD read cannot be issued early.** On a
machine where memory sits on a 3 GB/s SSD behind an unmeasured IOPS budget, that serialises the
slowest component in the system behind the fastest.

Measured cost per token (from `verify_decision.py` §1, at an assumed 300K IOPS):

| Design | Random reads/token | IO time | Prefetchable |
|---|---:|---:|---|
| (A) learned keys, top-k 128 × 8 layers *(my original spec)* | 1024 | 3.41 ms | no |
| (B) learned keys, shared pool, 3 layers *(Meta-corrected)* | 384 | 1.28 ms | no |
| (C) **n-gram hash**, 4 heads × 2 orders × 3 injection points | **24** | **0.08 ms** | **yes** |
| (D) **Apple block fetch**, 225M-param block per context | **1 sequential** | 15.0 ms *per context*, 0.015 ms/token amortised | **yes** |

**(C) issues 43× fewer reads than (A).** But the read count is the smaller half of the argument.

---

## 3. The decisive argument: robustness to the one number we have not measured

SSD random-read IOPS is the single hardware parameter this project has **not** grounded
([01](01_feasibility.md) lists it as UNVERIFIED). An L8 design does not bet the architecture on an
unmeasured quantity. So: how does each design behave as that assumption varies?

```
      IOPS   learned-key tok/s   hash tok/s   hash advantage
      500K               143.7        161.6            1.12x
      300K               133.9        161.6            1.21x
      100K                99.7        161.6            1.62x
       50K                72.1        161.6            2.24x
       20K                39.4        161.6            4.10x
```

**The hash design is completely flat at 161.6 tok/s across a 25× swing in IOPS.** The learned-key
design falls by 3.6× over the same range. Because the addresses are known ahead of time, the reads
hide entirely under compute, and the SSD stops being on the critical path at all.

This is the reason to choose it. Not that it is faster at the expected operating point — at 300K
IOPS the gain is a modest **1.21×** — but that **it removes an unmeasured hardware parameter from the
critical path entirely.** The learned-key design's viability depends on a number nobody has measured;
the hash design's does not.

Additional structural win: during **prefill**, every n-gram key for the whole prompt is known before
any computation starts, so all memory reads can be issued as one batched, sorted, near-sequential
pass — turning random IO into streaming IO. Prefill is the dominant user-visible latency on this box
(4.84 s for a 2048-token prompt, [01](01_feasibility.md)), so this matters more than the decode gain.

DeepSeek states this property directly: deterministic addressing "enables runtime prefetching from
host memory, incurring negligible overhead."

---

## 4. The second decision: block fetch over top-k gather

Apple fetches **one contiguous ~18M-parameter block per context** (scaled to our 2B backbone:
225M params = 45 MB), rather than gathering k scattered vectors per token.

| | top-k gather | block fetch |
|---|---|---|
| IO pattern | 384–1024 random reads **per token** | 1 sequential read **per context** |
| Cost | 1.28–3.41 ms/token | 15.0 ms per context = **0.015 ms/token** over a 1000-token generation |
| Bandwidth utilisation | terrible (4 KB pages for ~400-byte values) | near-peak sequential |

**~85× cheaper per token amortised.** The cost is granularity: a block commits you to one region of
memory for the whole context, so it captures topical/long-tail knowledge well and token-specific
lookup poorly.

That is why the recommendation is **both, in two tiers** — they fail in opposite directions.

---

## 5. Resolving the layer-placement contradiction

The literature genuinely conflicts:

- **Meta (Memory Layers at Scale):** ~**3** memory layers is the sweet spot; beyond that, replacing
  more FFN layers **degrades** performance — "sparse and dense layers are both needed and likely
  complementary."
- **ByteDance (UltraMemV2):** memory layers in **every** transformer block, and it works.

This is resolvable rather than contradictory. UltraMemV2 explicitly *redesigned* the mechanism to
make dense placement work — single-linear-projection value expansion, FFN-based value processing
borrowed from PEER, principled initialization, and a rebalanced memory-to-FFN compute ratio. Meta's
degradation was measured on Meta's formulation, where adding a memory layer **removes** an FFN layer;
UltraMemV2 rebalances rather than substitutes.

**Decision: start at ~3 access points over a shared pool (Meta's validated configuration, and the
conservative choice), and treat every-block placement as a Phase-3 experiment contingent on
reproducing UltraMemV2's value-processing redesign first.** Do not adopt the aggressive placement
and the aggressive scale simultaneously — that confounds two variables and makes failure
uninterpretable.

This corrects two concrete errors in [04](04_architecture.md): **8 memory layers** (should be 3) and
**unshared pools** (should be shared — Meta shares one pool across all access points, which keeps
parameter count fixed while multiplying access).

---

## 6. The hardest question: is one trillion parameters the right target?

**On the evidence: no.** This is the part of the original proposal that should change.

Three independent results all point the same way:

| Source | Finding |
|---|---|
| **UltraMemV2** | At equal activated compute, **2.5B/60B/top-768 beats 2.5B/120B/top-256** — 3× more *activation density* beat 2× more *total sparse parameters*. "Activation density has greater impact on performance than total sparse parameter count." |
| **DeepSeek Engram** | A **U-shaped scaling law** governs sparsity allocation between compute (MoE) and memory (Engram). U-shaped means an optimum exists and **past it, more memory is net negative.** |
| **Meta Memory Layers** | Performance degrades past ~3 memory layers as dense capacity is displaced. |

**"One trillion parameters" optimises the axis with the worst measured marginal return.** That is the
finding, and it is uncomfortable because it is the headline of the original proposal.

Where the proposal actually sits:

| | This proposal | Best published | Verdict |
|---|---:|---:|---|
| Memory : active ratio | 37.5 : 1 | 48 : 1 (UltraMemV2) | **within range — not the problem** |
| Absolute memory bank | 974 B | 128 B (Meta) | **7.6× beyond anything published** |

Note this corrects a claim I made earlier in this research: the *sparsity ratio* is not aggressive by
research standards. The **absolute scale** is.

### The staged target

Build in stages and let measurement decide where to stop:

| Stage | Memory params | Total | Disk @ ternary | vs. largest published | Storage |
|---|---:|---:|---:|---:|---|
| **S1** | 30 B | 56 B | 11.2 GB | 0.2× | fits 256 GB base |
| **S2** | 100 B | 126 B | 25.2 GB | 0.8× | fits 256 GB base |
| **S3** | 300 B | 326 B | 65.2 GB | 2.3× | fits 256 GB base |
| **S4** | 974 B | **1000 B** | 200.0 GB | 7.6× | **needs 512 GB / external NVMe** |

S2 on the 16 GB box, verified:

```
backbone 2B + experts 24B resident = 5.20 GB
memory 100B on SSD                 = 20.0 GB
TOTAL                              = 126B params, 25.2 GB disk
RAM: 5.20 weights + 0.54 KV + 0.50 activations = 6.24 GB of 9.66 GB budget
decode bandwidth limit             = 162 tok/s
hash-memory IO (prefetched)        = hidden under compute
```

**S1–S3 all fit the base 256 GB Mac mini.** Only S4 requires a hardware purchase. That means the
entire research programme — including the measurement that decides whether S4 is worth building —
runs on the machine already owned.

**Gate between stages:** fit the marginal-value curve of memory parameters at S1, S2, S3. If the
measured curve has turned over by S3, **stop and do not build S4** — the trillion would be
parameters that make the model worse. If it is still rising at S3, S4 is justified by measurement
rather than by ambition.

This is the honest way to keep the goal: **1T stays the target, but it must be earned.**

---

## 7. What updates in the main idea — and what does not

### Preserved (the premise, unchanged and now better supported)

> Intelligence emerges from dynamic patterns formed over a fixed computational substrate, not from
> continuously increasing parameter count.

Five independent labs now support this, and the strongest datum is one the original proposal did not
cite: **RETRO 7.5B matched GPT-3 175B — 25× fewer parameters — by moving knowledge out of weights.**
Google has shipped the idea twice (Gemma 3n, Gemma 4). The premise needs no defence.

### Updated

| Original proposal | Update | Why |
|---|---|---|
| Pattern Space via learned SDR/Hopfield addressing | **Deterministic n-gram hash addressing**, learned keys only in the semantic tier | Prefetchability; removes the re-indexing failure that killed PKM for a decade |
| 8 memory layers, separate pools | **Shared pool, ~3 access points** | Meta's measured sweet spot; sharing keeps params fixed while multiplying access |
| top-k gather from memory | **Two tiers: hash lookup + contiguous block fetch** | ~85× cheaper per token amortised on SSD |
| "1 trillion parameters" as the spec | **Staged 126B → 326B → 1T, gated on a measured U-curve** | Three independent diminishing-returns results |
| "Memory buys knowledge, not reasoning" (my own earlier risk statement) | **Downgraded** | Engram's reasoning gains (BBH +5.0) *exceeded* its knowledge gains (MMLU +3.4) |
| Weights ~ infrastructure, memory ~ knowledge | **Refined: dense and sparse are complementary, not substitutes** | Meta: "sparse and dense layers are both needed"; keep a real dense core |

### The one thing that gets *harder*

The original framing treated the dense backbone as mere plumbing. Every result says otherwise: Apple's
small LM is "an anchor capturing common knowledge and general reasoning"; Meta found dense and sparse
complementary; UltraMemV2 had to *rebalance toward* FFN compute. **The dense core is load-bearing and
should not be minimised to make room for memory.** This is the single biggest conceptual correction
to the original proposal.

---

## 8. Revised component mapping

| Original component | Revised implementation | Reference |
|---|---|---|
| 1. Dynamic Pattern Encoder | Keep BLT/entropy patching, but budget **~2× FLOP saving**, not 16× | [02](02_math_corrections.md) |
| 2. Pattern Space | Dense SSM/attention backbone — **enlarged, not minimised** | Apple, Meta, UltraMemV2 |
| 3. Associative Memory | **Tier A:** Engram n-gram hash (prefetched). **Tier B:** Apple hierarchical block fetch | §3, §4 |
| 4. Sparse Expert Selection | Fine-grained MoE, **fully RAM-resident** — no SSD expert traffic | [01 §6](01_feasibility.md) |
| 5. Pattern Interaction | Unchanged in intent; needs a falsifiable metric before it is buildable | [06](06_evaluation.md) |
| 6. Reasoning Engine | Test-time compute; ~271 s per 10⁴ thinking tokens at 37 tok/s | [01 §7](01_feasibility.md) |
| 7. Pattern Synthesis | Deferred — no published method at this scale; highest-risk component |  |
| 8. Memory Lifecycle | **Titans' gradient-based surprise metric**, not a hand-tuned decay constant | [03 E1](03_prior_art.md) |

---

## 9. Immediate next actions

1. **Verify licenses** on `deepseek-ai/Engram` and `apple/ml-memory-pretraining`. If both are
   restrictive, clean-room Engram from the paper — hash lookup is simple enough that this is
   genuinely tractable, which is itself an argument for the choice.
2. **Run `bench/`** on the Mac mini — especially SSD random-read IOPS, the parameter §3 shows the
   design is now insensitive to but which still sets the fallback envelope.
3. **Build S1 (56B) first.** It fits the current machine, requires no purchase, and produces the
   marginal-value curve that decides everything downstream.
4. **Do not buy the 512 GB machine yet.** S4 is the only stage that needs it, and S4 is gated on a
   measurement not yet taken.

---

## Sources

- [Conditional Memory via Scalable Lookup (Engram), DeepSeek-AI, arXiv:2601.07372](https://arxiv.org/abs/2601.07372) · [code](https://github.com/deepseek-ai/Engram)
- [Pretraining with hierarchical memories, Apple, ICLR 2026, arXiv:2510.02375](https://arxiv.org/abs/2510.02375) · [code](https://github.com/apple/ml-memory-pretraining)
- [UltraMemV2, arXiv:2508.18756](https://arxiv.org/abs/2508.18756) · [Ultra-Sparse Memory Network (ICLR 2025), arXiv:2411.12364](https://arxiv.org/abs/2411.12364)
- [Memory Layers at Scale, Meta, arXiv:2412.09764](https://arxiv.org/abs/2412.09764) · [code (CC-BY-NC)](https://github.com/facebookresearch/memory)
- [Gemma 3n model overview, Google](https://ai.google.dev/gemma/docs/gemma-3n) · [Gemma 4 Technical Report, arXiv:2607.02770](https://arxiv.org/html/2607.02770v1)
- [Large Memory Layers with Product Keys, arXiv:1907.05242](https://arxiv.org/abs/1907.05242)
- [Mixture of A Million Experts, arXiv:2407.04153](https://arxiv.org/abs/2407.04153)
- [Memory Grafting, arXiv:2605.20948](https://arxiv.org/abs/2605.20948)
