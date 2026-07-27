# 05 — Training and Memory Population

> Numbers from `verify_math.py` §10–§11.

---

## 1. The number that decides the programme

Training compute is `C ≈ 6·N·D` (forward 2ND + backward 4ND). The M4 GPU peaks at 4.4 TFLOP/s FP32;
at a realistic 30% MFU that is **1.32 TFLOP/s**.

| Scenario | FLOPs | Wall clock @ 30% MFU |
|---|---:|---:|
| 1T dense, Chinchilla-optimal (20 tok/param) | 1.20 × 10²⁶ | **2.88 × 10⁶ years** |
| 1T MoE, 5B active, 20T tokens | 6.00 × 10²³ | 1.44 × 10⁴ years |
| 1T MoE, 5B active, 1T tokens | 3.00 × 10²² | 7.20 × 10² years |
| 30B active, 1T tokens | 1.80 × 10²³ | 4.32 × 10³ years |
| 2B backbone, 100B tokens | 1.20 × 10²¹ | **28.8 years** |
| 500M backbone, 10B tokens | 3.00 × 10¹⁹ | 263 days |
| 100M backbone, 2B tokens | 1.20 × 10¹⁸ | **10.5 days** |

Even at an optimistic 50% MFU the top line is 1.73 × 10⁶ years. **This is not a scheduling problem
and no amount of engineering closes it.**

What actually fits in a wall-clock budget (30% MFU):

| Budget | Total FLOPs | N·D |
|---|---:|---:|
| 7 days | 7.98 × 10¹⁷ | 1.33 × 10¹⁷ |
| 30 days | 3.42 × 10¹⁸ | 5.70 × 10¹⁷ |
| 90 days | 1.03 × 10¹⁹ | 1.71 × 10¹⁸ |

At 30 days: a **100M** model gets 5.70B tokens (57 tok/param — above Chinchilla); a **1B** model gets
0.57B tokens (0.6 tok/param — hopelessly undertrained).

**Conclusion: the largest model that can be trained from scratch on this machine is ~100–300M
parameters.** The 2B backbone in [04](04_architecture.md) cannot be. That is a hard fact, and it
dictates everything below.

---

## 2. Therefore: download the backbone, don't train it

The dense backbone must come from open weights. Options, in order of preference:

1. **BitNet b1.58 2B4T** (Microsoft) — 2B params, 4T training tokens, **natively ternary**. This is
   the single best-fitting artifact in existence for this project: right size, right precision,
   trained natively rather than post-quantized (which matters — see [03 C2](03_prior_art.md)).
2. **Gemma 3n E2B / Gemma 4 E2B** — 5B raw / ~2B effective, and it **already implements PLE**, so
   the memory-injection plumbing exists and can be studied directly.
3. A 2B open dense model, adapted toward ternary — accepting the quantization degradation that
   *Scaling Laws for Precision* predicts will be **worse** the more tokens the checkpoint was
   trained on.

**Gate G2** in [07](07_kill_switches.md) measures that degradation before committing.

---

## 3. The key asymmetry: writing costs a forward pass, training costs six

This is the whole reason the project is viable.

**Regime (a) — gradient-train the memory values** (Memory Layers at Scale style):

| Tokens per memory param | FLOPs | Wall clock |
|---|---:|---:|
| 1 | 5.69 × 10²⁴ | **1.37 × 10⁵ years** |
| 20 | 1.14 × 10²⁶ | 2.73 × 10⁶ years |

**Dead by 5–6 orders of magnitude.**

**Regime (b) — write the values by encoding a corpus once** (RETRO / Memory Grafting style). Cost is
`2 · N_encoder · corpus_tokens`, forward only, one epoch:

| Encoder | Chunk | Corpus tokens | FLOPs | Wall clock |
|---:|---:|---:|---:|---:|
| 10 M | 64 | 3.04 × 10¹⁰ | 6.09 × 10¹⁷ | **5.3 days** |
| 10 M | 128 | 6.09 × 10¹⁰ | 1.22 × 10¹⁸ | 10.7 days |
| 30 M | 64 | 3.04 × 10¹⁰ | 1.83 × 10¹⁸ | **16.0 days** |
| 30 M | 128 | 6.09 × 10¹⁰ | 3.65 × 10¹⁸ | 32.0 days |
| 100 M | 64 | 3.04 × 10¹⁰ | 6.09 × 10¹⁸ | 53.4 days |
| 500 M | 256 | 1.22 × 10¹¹ | 1.22 × 10²⁰ | 1068 days |

**Correction to an earlier draft of this research:** I initially wrote that populating ~2×10⁹ slots
was "a weeks-scale job" while the same script printed 4489 days. At a 500M encoder and 256-token
chunks it is **12.3 years, not weeks.** The job is feasible *only* with a small encoder and short
chunks. The viable configurations are the bolded rows.

**A ~10⁵-year problem becomes a ~2-week problem purely by writing instead of training.** That
substitution is the load-bearing claim of this project.

Corpus availability is not a constraint: 30.4B tokens at 64-token chunks, against the Pile (~300B)
or FineWeb (~15T).

---

## 4. Memory Grafting makes this a published technique, not an invention

*[arXiv:2605.20948](https://arxiv.org/abs/2605.20948)* independently validates regime (b):

- Run a **grafting model offline** over frequent local n-grams
- Store **final-token hidden representations as memory values**
- The recipient model retrieves them by **exact longest-match suffix lookup** — O(1), no learned keys
- Lightweight **projections and gates** adapt the retrieved memory; a hash-based Engram fallback
  covers unmatched contexts
- At 2.8B scale: **53.86** average benchmark score vs **51.95** (MoE) and **52.43** (vanilla Engram)

This is close enough to the plan that it should be treated as the reference implementation of the
write path, not as related work.

---

## 5. What actually gets trained locally

Only three things, all small:

| Artifact | Params | Method | Cost |
|---|---:|---|---|
| Memory projections & gates | ~10–50 M | gradient, backbone frozen | days |
| Router for the MoE pool | ~10 M | gradient | hours |
| Fine-grained experts | 24 B pool, trained **one at a time** | **c-BTM** | weeks, resumable |

**c-BTM is the only published training paradigm that decomposes into single-machine-sized pieces:**
k-means the corpus into domains, train one expert per cluster **independently** with no
synchronisation, sparsely activate a subset at inference. A Mac mini is a one-node cluster; c-BTM is
what makes that a coherent position rather than a limitation. **Sparse Upcycling** initialises each
expert from the downloaded dense checkpoint rather than from scratch.

Every expert is an independent, resumable job. The machine can be interrupted, rebooted, and
restarted without losing more than one expert's progress.

---

## 6. The assembled programme

| Phase | What | Method | Time |
|---|---|---|---|
| 0 | Measure the machine | `bench/` | 1 day |
| 1 | Acquire backbone | download BitNet b1.58 2B4T | hours |
| 2 | Write Tier-A hash memory | offline encode, 10–30M grafting model, 64-tok chunks | **5–16 days** |
| 3 | Write Tier-B block memory | k-means cluster + block encode | days |
| 4 | Train projections/gates | gradient, backbone frozen | days |
| 5 | Build expert pool | c-BTM + sparse upcycling, one expert at a time | weeks |
| 6 | Measure the U-curve | S1 → S2 → S3 | weeks |

**Nothing in this programme requires a GPU cluster, and nothing takes longer than a few weeks.** The
trillion-parameter target is reachable on a Mac mini because ~97% of the parameters are written once
rather than trained repeatedly — and because the stages that would push past 300B are gated on
measurement rather than assumed ([09 §6](09_method_comparison_and_decision.md)).

---

## 7. Risks specific to training

1. **Quantization/pretraining interaction.** *Scaling Laws for Precision*: PTQ degradation grows with
   pretraining tokens. Downloaded checkpoints are heavily trained. Prefer natively-ternary BitNet.
2. **Written memory ≠ trained memory.** Every published memory result *trained* its values. Writing
   them is cheaper and is validated by RETRO and Memory Grafting, but not at 974B. This is gate **G3**.
3. **c-BTM experts may not compose.** Independently trained experts can be mutually inconsistent;
   BTM's merge step is the mitigation and must be measured, not assumed.
4. **Encoder quality bounds memory quality.** A 10M grafting model writes worse values than a 500M
   one, and the 100× cost difference is exactly the trade to measure at S1.
