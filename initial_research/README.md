# Initial Research — Can a 1-Trillion-Parameter Intelligent System Run on a 16 GB M4 Mac mini?

**Status:** research complete, no code written yet.
**Every number in these documents is produced by [`verify_math.py`](verify_math.py).**
Output is committed verbatim at [`verify_math_output.txt`](verify_math_output.txt). Nothing here is hand-arithmetic.

```
$ python3 verify_math.py    → exit 0
```

---

> ## ⚠️ ARCHITECTURE REVERSED — read [10_dynamic_substrate.md](10_dynamic_substrate.md) first
>
> Docs 01–09 optimised a premise that had been quietly replaced: they built an **external memory
> bank** (storage + lookup) instead of a **fixed dynamic substrate** (patterns + interaction), which
> is what the original proposal actually claims.
>
> One measurement reversed it. Allen-Zhu & Li: **2 bits per parameter**, and a 7B model already
> exceeds English Wikipedia + textbooks. A fixed brain in **6.5 GB of RAM holds 52 Gbit — 3.7×
> Wikipedia+textbooks — at 84 GB/s.** The old design put 97% of parameters on SSD at 3 GB/s: **the
> tier that is 28× slower**, to solve a capacity problem that mostly did not exist.
>
> **Now:** dynamics-first, everything in RAM, external store is a *conditional overflow tier*.
> Disk drops from 200 GB to ~5–20 GB, no hardware purchase is needed, and time-to-first-signal
> drops from 3–6 months to **~2 weeks**.
>
> Docs 01–03 (feasibility, math, prior art) remain valid. 04 is rewritten. 09 is authoritative on
> *mechanism*, superseded on *tier*.

## The verdict, in one paragraph

**Yes — but only if ~97% of those parameters are *written* rather than *trained*, and only if "one
trillion" is treated as an outcome to be earned rather than a specification.** A literal 1T dense
model is off by ~200× on storage and ~10⁶ on training time. A literal port of an existing 1T MoE
(Kimi K2) runs at **0.47–4.7 tok/s** here — technically "running," practically useless. What *does*
close is a staged system reaching 1.000 × 10¹² parameters partitioned as **26B trained + 974B
written**, 200 GB on SSD, 6.3 GB of the ~9.7 GB usable RAM, at an estimated **~37 tok/s**.

The load-bearing insight: **gradient-training 974B parameters takes ~10⁵ years on this hardware;
writing them with one forward pass each takes ~16 days.** That is the whole difference between
impossible and a two-week batch job.

**The premise is no longer speculative — five major labs have now shipped versions of it, and four
released code.** Google ships it in production (Gemma 3n → Gemma 4 Per-Layer Embeddings); DeepSeek,
Meta, ByteDance and Apple have all published memory-augmented systems that beat compute-matched MoE.
See [08](08_prior_attempts_and_code.md).

**But the research also refutes the headline number.** Three independent results — UltraMemV2's
controlled experiment, DeepSeek's U-shaped scaling law, and Meta's layer-count finding — agree that
**total sparse parameter count is the axis with the worst marginal return**. So the plan is now
staged (126B → 326B → 1T) with continuation gated on a measured curve. The premise survives; the
target must be earned. See [09](09_method_comparison_and_decision.md).

---

## Read in this order

| # | Document | What it settles |
|---|---|---|
| 01 | [Feasibility & hardware envelope](01_feasibility.md) | What the machine physically permits. Storage, bandwidth, roofline. Why porting Kimi K2 fails. |
| 02 | [Math corrections to the root README](02_math_corrections.md) | Three quantitative claims in `/README.md` are wrong or misleading. One by 52.8 orders of magnitude. |
| 03 | [Prior art the proposal is missing](03_prior_art.md) | Lines of work absent from the proposal, including the two that make it possible. |
| **08** | **[Who tried this, what code exists, what they learned](08_prior_attempts_and_code.md)** | **Five labs, four codebases. Google's PLE in production. Why product-key memory stalled for a decade — and what fixed it.** |
| 09 | [Method comparison and decision](09_method_comparison_and_decision.md) | Seven memory mechanisms compared. Authoritative on *mechanism*; superseded on *tier* by 10. |
| **10** | **[The dynamic substrate](10_dynamic_substrate.md)** | **THE REVERSAL. Capacity math, the BDH/TTT/Hopfield lineage, and the 2-week experiment that decides the whole programme.** |
| 04 | [Architecture specification](04_architecture.md) | The dynamics-first design. *Rewritten per 10.* |
| 05 | [Training & memory population](05_training_and_population.md) | What can be trained here (little), downloaded (the backbone), and written (almost all). |
| 06 | [Evaluating intelligence](06_evaluation.md) | How to compare against dense baselines — and against RAG — without fooling ourselves. |
| 07 | [Kill switches](07_kill_switches.md) | Numeric, falsifiable abort conditions. Three of six produce a publishable negative when they fail. |

Microbenchmarks that **must be run on the actual Mac mini** before anything is built:
[`bench/`](bench/) — see [`bench/README.md`](bench/README.md).

Verification scripts: [`verify_math.py`](verify_math.py) → [output](verify_math_output.txt);
[`verify_decision.py`](verify_decision.py) → [output](verify_decision_output.txt).

---

## The headline numbers

Hardware envelope (M4 Mac mini, 16 GB):

| Quantity | Value | Source |
|---|---|---|
| Unified memory | 16 GiB = 17.18 GB | Apple spec |
| Memory bandwidth (peak) | 120 GB/s | Apple spec |
| Memory bandwidth (decode-effective, assumed 70–85%) | 84–102 GB/s | assumption, **must be measured** |
| Internal SSD sequential read | ~3 GB/s | third-party benchmarks |
| GPU FP32 peak | 4.4 TFLOP/s | third-party benchmarks |
| GPU effective @ 30% MFU | 1.32 TFLOP/s | assumption, **must be measured** |
| Engine RAM budget after macOS | ~9.66 GB | derived |

The 1T system that closes:

| Component | Params | Footprint | Tier |
|---|---:|---:|---|
| Dense backbone (SSM + sparse attention) | 2.000 B | 0.400 GB | RAM |
| Fine-grained MoE pool (all resident) | 24.000 B | 4.800 GB | RAM |
| Product-key memory values | 973.955 B | 194.791 GB | SSD |
| Product-key index | 0.045 B | 0.089 GB | RAM |
| **Total** | **1.000000 × 10¹²** | **200.0 GB** | |

Per-token limits — the binding one is bandwidth, not compute:

```
[bandwidth] 0.520 GB/token / 84 GB/s      =  6.19 ms → 161.5 tok/s
[compute  ] 5.20 GFLOP/token / 1.32 TFLOP/s =  3.94 ms → 253.8 tok/s
[memory IO] 1024 random reads / 300K IOPS =  3.41 ms → 293.0 tok/s
                                  serialized → 73.8 tok/s
                    plan against 50% of that → 36.9 tok/s
```

Prefill, not decode, is the latency users will feel: **4.84 s for a 2048-token prompt** (423 tok/s).

---

## What is genuinely novel here, and what is not

Not novel — all published, all with released code ([08](08_prior_attempts_and_code.md)): memory
layers, product keys, n-gram conditional memory, hierarchical memory banks, ternary weights, MoE,
Mamba. **The core premise is now mainstream research**, not a contrarian bet.

What remains genuinely untested:

1. **The written-vs-trained memory gap at scale.** Every published memory system *trained* its
   values. This project *writes* them. RETRO and Memory Grafting validate the write path, but not at
   974B. **Nobody has published this number** — measuring it is a real contribution regardless of
   which way it comes out. This is gate **G3**.
2. **Two-tier memory (hash lookup + block fetch).** Engram and Apple each validated one; nobody has
   combined them. They may prove redundant rather than complementary.
3. **Absolute scale.** 974B is **7.6× beyond the largest published memory bank** (Meta's 128B). Note
   the *sparsity ratio* (37.5:1) is **not** aggressive — UltraMemV2 runs 48:1. The extrapolation is
   in size, not sparsity.

[07](07_kill_switches.md) is designed to kill each of these fast if false.

---

## The honest risks

**1. The trillion may be the wrong goal.** Three independent findings say total sparse parameters is
the lowest-return axis. UltraMemV2's controlled result is the sharpest: at equal compute,
**2.5B/60B/top-768 beat 2.5B/120B/top-256** — tripling activation density beat doubling total
parameters. If the U-curve turns before S3, this ships as a 126B or 326B system and the trillion is
abandoned on evidence. That is a designed outcome, not a failure.

**2. Reasoning.** Reasoning scales with active compute, hard-bounded here at ~2.6B active parameters,
and must be bought back with test-time compute — at ~37 tok/s, 10⁴ thinking tokens costs **271 s per
answer**. Energy is irrelevant (≈1 Wh per 2-minute answer); wall-clock is the cost.

> **Correction to an earlier draft of this research:** I previously stated flatly that
> "parameters-as-memory buys knowledge, not reasoning." DeepSeek's Engram results contradict that —
> its *reasoning* gains (BBH +5.0, ARC-C +3.7) **exceeded** its knowledge gains (MMLU +3.4), with the
> proposed mechanism being that offloading local dependencies to lookup frees attention capacity for
> global context. The risk is downgraded from "expected" to "open".

**3. This may just be RAG.** If a 2.6B dense model with a conventional retrieval pipeline over the
same corpus matches the system, the architecture is an expensive reimplementation of retrieval. That
comparison (Arm E in [06](06_evaluation.md)) must be run **first**, not last.

---

## Open assumptions requiring measurement before Phase 1

These are assumptions, not results. They are listed again as gate G0 in [07](07_kill_switches.md).

1. Decode-effective memory bandwidth is 70–85% of the 120 GB/s peak.
2. Sustained GPU MFU on MLX reaches 30% for training-shaped workloads.
3. SSD random-read IOPS at 4 KB depth ≥ 100K.
4. `iogpu.wired_limit_mb` can be raised to ~12 GiB without destabilising macOS.
5. Ternary (1.6 bit/param) kernels achieve ≥50% of fp16 bandwidth efficiency on Metal.

If (1) or (3) are materially worse than assumed, the throughput estimate degrades roughly linearly
and the design point in [04](04_architecture.md) must be re-derived, not patched.
