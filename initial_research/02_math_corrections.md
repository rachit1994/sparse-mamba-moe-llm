# 02 — Corrections to the Quantitative Claims in `/README.md`

> Reproduced by `verify_math.py` §1–§3.

Three numerical claims in the root proposal are wrong or materially misleading. One of them is the
headline capacity argument. They are corrected here rather than silently patched, because the
*direction* of two of the errors flatters the proposal, and a proposal that flatters itself is not
worth building against.

---

## Correction 1 — SDR capacity is understated by 52.8 orders of magnitude

**README claims:**
```
10,000 units, 100 active
C(10000,100) ≈ 10^189
```

**Correct value:**

```
C(10000,100) exact digits      : 242
C(10000,100) log10             : 241.8143
cross-check via lgamma (log10) : 241.8143  → agrees
```

**C(10000, 100) ≈ 10²⁴¹·⁸, not 10¹⁸⁹.** The claim is off by 52.8 orders of magnitude — in the
*conservative* direction, which is why it went unnoticed.

Where 10¹⁸⁹ might have come from — none of these are the stated configuration:

| Configuration | log₁₀ |
|---|---:|
| C(10000, 60) | 158.00 |
| C(10000, 64) | 166.81 |
| C(10000, 70) | 179.82 |
| C(10000, 80) | 201.01 |
| C(10000, 100) | **241.81** |

Anchors for intuition:

- Numenta's canonical SDR, C(2048, 40) ≈ 10⁸⁴·⁴
- Atoms in the observable universe ≈ 10⁸⁰

### The deeper error: the unit, not the arithmetic

Fixing 10¹⁸⁹ → 10²⁴¹·⁸ corrects the arithmetic and misses the real problem. **C(10000,100) is a
count of patterns — an address space. Its logarithm is the storage capacity.**

```
C(10000,100)        = 10^241.8   ← COUNT of distinguishable patterns
log2(C(10000,100))  = 803 bits   ← what one pattern can actually CARRY
```

The hard ceiling for any n-unit substrate is **n bits**, regardless of encoding:

| Units n | Active w | log₂ C(n,w) | Raw ceiling |
|---:|---:|---:|---:|
| 10,000 | 100 | 803 bits | 10,000 bits |
| 100,000 | 1,000 | 8,073 bits | 100,000 bits |
| 1,000,000 | 10,000 | 80,785 bits | 1,000,000 bits |

**Why the correction does not sink the premise.** Restated in bits, the argument still works — it
just needs the right number. A 6.5 GB substrate holds 52 Gbit, and Allen-Zhu & Li measured that
English Wikipedia + textbooks is ~14 Gbit. **The fixed brain has 3.7× headroom for Wikipedia-scale
knowledge.** See [10_dynamic_substrate.md](10_dynamic_substrate.md) — this number reversed the
architecture decision.

The capacity argument proves *address space* is not the bottleneck, which was never in dispute. The
binding quantity is **bits**, and stated in bits the premise survives.

**Recommendation:** keep the corrected number, but demote the argument from "capacity argument" to a
footnote establishing that representational addressing is not the limiting factor. Replace the load
bearing claim with a *retrieval* bound — how many patterns can be recovered at a given noise level —
which is what modern Hopfield theory actually provides and what can be measured.

---

## Correction 2 — the "16×" attention saving is a sub-term, not a system saving

**README claims:**
```
Transformer complexity O(n²)
Reducing sequence length by 4× yields approximately 16× less attention compute.
```

The first sentence is incomplete and the second is true only of a minority term.

Per token per layer, a transformer costs approximately:
- **linear terms** (QKV, output projection, MLP): `12·d²` MACs — scales with *number of tokens*
- **attention terms** (scores, weighted values): `2·n·d` MACs — scales with *n × number of tokens*

So attention's share is `n / (6d)`, and cutting n by 4× cuts the linear term by 4× (fewer tokens) and
the attention term by 16×:

| d | n | Attention share of FLOPs | Attention-only reduction | **Total model reduction** |
|---:|---:|---:|---:|---:|
| 2048 | 2048 | 14.3% | 16.00× | **4.48×** |
| 4096 | 4096 | 14.3% | 16.00× | **4.48×** |
| 4096 | 16384 | 40.0% | 16.00× | **5.71×** |
| 8192 | 131072 | 72.7% | 16.00× | **8.80×** |

**The end-to-end saving at realistic shapes is ~4.5×, not 16×** — and that is before adding back the
byte-level encoder/decoder that any patching scheme requires. Meta's Byte Latent Transformer, which is
the method the README proposes for this, reports **~50% inference FLOP savings** at 6–8 byte average
patches — i.e. ~2×, consistent with this analysis once encoder overhead is included, and far from 16×.

**Recommendation:** state the claim as "~2× measured (BLT), ~4.5× theoretical ceiling for the
attention-and-linear split, 16× on the attention term alone." Cite BLT's measured number as the
planning figure.

---

## Correction 3 — the MoE "32×" is arithmetically right and rhetorically wrong

**README claims:**
```
64 experts × 4M = 256M total; top-2 → 8M active; 256M/8M = 32× less expert compute
```

The arithmetic checks out: 32×. The framing does not, for three reasons.

1. **Memory is unchanged.** All 256M parameters must be resident or streamable. On a
   memory-constrained box this is the *only* number that matters, and sparsity does not improve it.
   This is precisely why Kimi K2 fails on this machine ([01 §5](01_feasibility.md)).
2. **Only the expert sub-graph is reduced.** Attention, router, embeddings, and norms are not.
3. **Amdahl's law applies immediately:**

| Expert share of FLOPs | End-to-end speedup from 32× expert sparsity |
|---:|---:|
| 50% | 1.94× |
| 60% | 2.39× |
| 70% | 3.11× |
| 80% | 4.44× |

**A 32× sparsity ratio buys between 1.9× and 4.4× end-to-end.** The README's "32×" invites the reader
to budget for the former.

**Recommendation:** report MoE sparsity as two separate numbers — *parameter ratio* (32×, a memory
and knowledge-capacity statement) and *end-to-end speedup* (~2–4×, a latency statement). Conflating
them is how MoE proposals overpromise.

---

## Non-numerical issues worth flagging

**The decay model is unfalsifiable as written.** `U(t) = U₀e^(-λt)` has no stated λ, no units, and no
measurement procedure. Either give λ a value and an experiment that fits it, or drop the formula and
say "memories decay; the schedule is a hyperparameter."

**`Score = Confidence × Usage × Importance` is under-specified.** Three undefined quantities on a
multiplicative scale means any one going to zero deletes the memory. That is probably wrong for
`Usage` (a correct, never-yet-needed fact should not be evicted). Specify ranges and floors.

**`Intelligence ∝ Parameters` is a strawman as stated.** The scaling literature does not claim
parameters alone; Chinchilla's result is that parameters and *data* must scale together, and the
compute-optimal ratio is ~20 tokens/param. The proposal's real opponent is `Intelligence ∝ Compute`,
which is a much stronger claim and the one worth arguing against. Framing the opponent accurately
matters, because the trained/written split in [05](05_training_and_population.md) is exactly an
argument about compute, not about parameters.

---

## What survives

The corrections above weaken three supporting arguments. They do **not** touch the central hypothesis,
which is independently supported by evidence the proposal does not cite:

- **RETRO** (7.5B params + 2T-token retrieval database) matched GPT-3 175B on the Pile — a **25×
  parameter reduction** achieved by moving knowledge out of weights. This is the strongest existing
  evidence for "weights are infrastructure, patterns/memory are knowledge."
- **Memory Layers at Scale** (Meta, 2024) showed memory-augmented models beating dense models with
  **>2× the compute budget**, with >100% accuracy gains on factual QA.
- **Scaling Monosemanticity** (Anthropic, 2024) extracted up to 34M interpretable features from a
  production model's residual stream — direct empirical evidence that concepts *are* sparse
  activation patterns over a fixed substrate.

The hypothesis is defensible. The arithmetic supporting it needed to be right first.
See [03_prior_art.md](03_prior_art.md).

---

## Sources

- [Byte Latent Transformer: Patches Scale Better Than Tokens, arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
- [Improving language models by retrieving from trillions of tokens (RETRO), arXiv:2112.04426](https://arxiv.org/pdf/2112.04426)
- [Memory Layers at Scale, arXiv:2412.09764](https://arxiv.org/abs/2412.09764)
- [Scaling Monosemanticity, Anthropic](https://arxiv.org/abs/2605.29358)
- [Training Compute-Optimal Large Language Models (Chinchilla), arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
