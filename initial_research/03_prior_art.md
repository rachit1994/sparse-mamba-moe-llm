# 03 — Prior Art the Proposal Is Missing

The root proposal cites Hopfield, Mamba, Switch/Mixtral/DeepSeek-MoE, HTM, AlphaZero, RAG/RETRO,
and JEPA. That is a reasonable map of the *idea space*. It omits most of the work that determines
whether the idea is **buildable on this machine**.

Below: 14 lines of work, what each contributes, and what it changes about the plan. The two marked
**LOAD-BEARING** are the ones without which the trillion-parameter target is unreachable.

---

## A. The two that make 10¹² parameters possible

### A1. Mixture of a Million Experts / PEER — **LOAD-BEARING**
*Xu Owen He, Google DeepMind, 2024 ([arXiv:2407.04153](https://arxiv.org/abs/2407.04153))*

Product-key retrieval over **more than a million tiny experts**. Splits queries and keys into
sub-components, reducing top-k retrieval from `O(N·d)` to `O((√N + k²)·d)`.

**Why this is the unlock.** A naive router over N experts costs O(N·d) per token — for N = 4.76×10⁸
that is ~10¹² operations per token, impossible. Product keys make the *index* O(√N):

| Slots N | √N | Key params | Key index size (fp16) | Value params @ d=2048 |
|---:|---:|---:|---:|---:|
| 10⁶ | 1,000 | 2.0 M | 4.1 MB | 2.0 B |
| 10⁸ | 10,000 | 20.5 M | 41.0 MB | 204.8 B |
| **4.76×10⁸** | **21,817** | **44.7 M** | **89.4 MB** | **974.8 B** |
| 10⁹ | 31,623 | 64.8 M | 129.5 MB | 2048.0 B |

**974 billion addressable parameters are indexed by 89 MB of RAM.** That single line is why the
trillion-parameter target is arithmetically reachable at all. It is absent from the proposal.

*Precursor:* Lample et al., *Large Memory Layers with Product Keys* (2019) — the original product-key
memory formulation.

### A2. Memory Layers at Scale — **LOAD-BEARING**
*Meta FAIR, Dec 2024 ([arXiv:2412.09764](https://arxiv.org/abs/2412.09764), ICML 2025)*

Trainable key-value memory layers that add parameters **without adding FLOPs**. Scaled to **128B
memory parameters** against backbones up to 8B, pretrained to 1T tokens — two orders of magnitude
beyond prior work.

Results: memory-augmented models **outperform dense models with more than 2× the compute budget**,
and beat MoE models matched for both compute and parameters. Gains are **>100% on factual QA**
(NaturalQuestions, TriviaQA).

**What it establishes:** the proposal's core hypothesis is not speculative — memory parameters
demonstrably substitute for compute on knowledge tasks at scale. **What it warns:** the gains are
concentrated on *factual* tasks. This is the empirical basis for the "knowledge-rich,
reasoning-poor" risk in [01](01_feasibility.md) and [06](06_evaluation.md).

**What this proposal does differently:** Memory Layers *trains* its memory values. At 974B params
that costs ~1.4×10⁵ years here ([05](05_training_and_population.md)). This proposal *writes* them.
That substitution is the actual research contribution and the thing most likely to be wrong.

---

## B. Running models larger than RAM (all absent from the proposal)

### B1. LLM in a Flash
*Apple, Dec 2023 ([arXiv:2312.11514](https://arxiv.org/abs/2312.11514), ACL 2024)*

Directly on-point: Apple hardware, flash storage, DRAM-constrained. Two techniques — **windowing**
(reuse recently activated neurons) and **row-column bundling** (larger contiguous flash reads).
Runs models **2× the size of available DRAM**, with **4–5× speedup on CPU and 20–25× on GPU** versus
naive loading.

Critically, it builds an explicit **inference cost model of flash characteristics** — which is exactly
the discipline `bench/` is meant to reproduce for the M4's SSD before any code is written.

### B2. Active-weight swapping between DRAM and flash
*[arXiv:2504.08378](https://arxiv.org/abs/2504.08378), Apr 2025* — successor line to B1. *(Full text
returned HTTP 403; summarized from search metadata only. **UNVERIFIED** — read before relying on it.)*

### B3. MoE offloading with LRU expert caching
*Eliseev & Mazur, 2023 ([arXiv:2312.17238](https://arxiv.org/abs/2312.17238))*

Key empirical finding: **MoE expert activation is temporally local** — some experts stay active for
2–4 token runs. This motivates LRU caching plus speculative prefetch, enabling Mixtral-8x7B on
consumer hardware.

**Why it does not rescue Kimi K2 here:** Mixtral is 8 experts / 47B total; K2 is 384 experts / 1000B.
The cache-to-pool ratio is ~30× worse ([01 §5](01_feasibility.md)). Expert locality is real but
insufficient at K2's granularity.

### B4. PowerInfer / PowerInfer-2
*SJTU, 2023–2024* — hot/cold neuron locality; PowerInfer-2 runs 47B models on smartphones. Same
principle as B1 applied to neuron-level rather than expert-level sparsity.

### B5. DiskANN / SPANN — SSD-resident billion-scale ANN
*Microsoft, NeurIPS 2019*

Indexes and serves a **billion-point** database on a **single node with 64 GB RAM and one SSD**:
>5000 QPS, <3 ms mean latency, 95%+ 1-recall@1, where FAISS and IVFOADC+G+P plateau near 50% recall
at comparable memory.

**Relevance:** DiskANN is the fallback if product keys prove insufficient for the memory layer — and
the reference implementation for the SSD access patterns the memory store needs. Note the RAM gap:
DiskANN assumes 64 GB, we have ~9.7 GB. **This is why product keys (89 MB index) are preferred over
ANN (tens of GB of PQ codes) — see A1.**

---

## C. Making parameters small (partly present)

### C1. BitNet b1.58 / BitNet b1.58 2B4T
*Microsoft, 2024–2025 ([arXiv:2504.12285](https://arxiv.org/html/2504.12285v2))*

The first open-source **native 1-bit LLM at 2B scale**, trained on 4T tokens. Ternary weights
{−1, 0, +1} via absmean quantization. Measured: **0.4 GB non-embedding memory vs 1.4–4.8 GB** for
fp16 peers; **0.028 J vs 0.186–0.649 J** per token; **29 ms vs 41–124 ms** CPU latency.
`bitnet.cpp` reports 2.37–6.17× speedups with 71.9–82.2% energy reduction on x86.

**Crucial distinction the proposal must respect: BitNet is trained *natively* ternary, not
post-training quantized.** Post-hoc ternarization of an fp16 model does not reach these numbers.

### C2. Scaling Laws for Precision — **the risk nobody mentions**
*Kumar et al., Nov 2024 ([arXiv:2411.04330](https://arxiv.org/abs/2411.04330))*

Fit on 465+ pretraining runs. Central finding: **post-training-quantization degradation *increases*
with the number of pretraining tokens** — eventually additional pretraining data becomes actively
harmful if you intend to quantize. Low precision reduces "effective parameter count."

**Direct threat to this project.** The plan depends on ternary weights *and* on heavily-trained
starting checkpoints (downloaded open weights trained on 15T+ tokens). Those two choices interact
badly, in exactly the regime this paper measures. Mitigation: train natively ternary (C1) or accept
measured degradation. This becomes gate **G2** in [07](07_kill_switches.md).

---

## D. Training without a datacenter (absent from the proposal)

### D1. Branch-Train-Merge / c-BTM
*BTM: [arXiv:2208.03306](https://arxiv.org/abs/2208.03306); c-BTM: Gururangan et al., 2023*

**Embarrassingly parallel** expert training: k-means-cluster the corpus into domains, train one
expert per cluster **independently**, sparsely activate a subset at inference. Eliminates nearly all
multi-node synchronization.

**Why this matters more here than anywhere else:** a Mac mini is a one-node cluster. c-BTM is the
only training paradigm in the literature that decomposes into pieces small enough to train *serially,
one at a time, resumably* on a single machine — and merge into something larger. It is the training
plan for the MoE pool in [05](05_training_and_population.md).

### D2. Sparse Upcycling
*Google, 2023* — initialize an MoE from a trained dense checkpoint. Combined with D1, lets the expert
pool start from downloaded weights rather than from scratch.

### D3. Scaling Laws for Fine-Grained Mixture of Experts
*Ludziejewski, Krajewski et al., ICML 2024 ([arXiv:2402.07871](https://arxiv.org/abs/2402.07871))*

Introduces **granularity** as a hyperparameter and shows the standard practice of sizing experts to
match the FFN layer is **not optimal at almost any compute budget**. Finer granularity improves the
frontier, and the dense-vs-MoE efficiency gap *widens* with scale.

**This is the quantitative justification for the "fine-grained" choice in
[04](04_architecture.md)** — it is not an aesthetic preference, it is on the measured optimal frontier.

---

## E. Memory & reasoning at test time (absent from the proposal)

### E1. Titans: Learning to Memorize at Test Time
*Behrouz et al., Google Research, Dec 2024 ([arXiv:2501.00663](https://arxiv.org/pdf/2501.00663), NeurIPS 2025)*

Neural long-term memory module that memorizes at **test time**, scaling past 2M-token context. Uses a
**surprise metric** — the gradient of the loss — to decide what is novel enough to store, with
momentum over past surprise and a gated adaptive forgetting mechanism.

**This is a rigorous, published version of the proposal's Components 3 and 8** (Associative Memory and
Memory Lifecycle Manager). The proposal's `U = Confidence × Usage × Recency` is a hand-specified
heuristic; Titans' gradient-based surprise signal is learned and measurable. **Adopt the surprise
formulation** and cite it, rather than inventing a decay constant ([02](02_math_corrections.md)).

### E2. RETRO
*DeepMind, 2021 ([arXiv:2112.04426](https://arxiv.org/pdf/2112.04426))*

Cited in the proposal, but its *headline number* is not: **RETRO 7.5B matches GPT-3 175B on the Pile
— 25× fewer parameters** — by retrieving from a 2-trillion-token database.

This is the single strongest published evidence for the proposal's central hypothesis, and it should
be the headline of the executive summary rather than a bullet in a reference list. It also supplies
the chunk-size convention (64 tokens) used in the memory-population math in
[05](05_training_and_population.md).

### E3. Scaling Monosemanticity / Toy Models of Superposition
*Anthropic, 2022–2024 ([arXiv:2605.29358](https://arxiv.org/abs/2605.29358))*

Sparse autoencoders extracted **1M, 4M, and 34M interpretable features** from Claude 3 Sonnet's
residual stream. Features are multilingual, multimodal, abstract, and **causally steerable**.

**Direct empirical support for the proposal's Fundamental Hypothesis** — concepts really are sparse
activation patterns over a fixed substrate, and they can be enumerated and manipulated. It also
supplies a *measurement instrument*: train an SAE on the backbone's residual stream and count
recoverable features. That is the only concrete way found to make "pattern formation" falsifiable,
and it becomes gate **G4** in [07](07_kill_switches.md).

### E4. Test-time compute scaling
*o1/o3-class systems; s1: simple test-time scaling, 2025* — budget forcing, repeated sampling. The
mechanism by which a ~37 tok/s system buys back reasoning quality it cannot get from parameters.

### E5. On the Measure of Intelligence
*Chollet, 2019 ([arXiv:1911.01547](https://arxiv.org/abs/1911.01547))*

> "The intelligence of a system is a measure of its skill-acquisition efficiency over a scope of
> tasks, with respect to priors, experience, and generalization difficulty."

The proposal asks "how would we compare intelligence" without defining it. This is the definition to
use, and it has a sharp consequence: **a system whose knowledge lives in a written memory store has
enormous "experience," so raw benchmark skill overstates its intelligence.** The evaluation protocol
in [06](06_evaluation.md) is built to control for exactly that.

---

## F. Also relevant, lower priority

| Work | Contribution | Where it lands |
|---|---|---|
| Byte Latent Transformer (Meta, 2024) | Entropy-based dynamic patching; ~50% inference FLOP saving | Cited in proposal; corrected number in [02](02_math_corrections.md) |
| Mamba / Mamba-2 (Gu & Dao) | O(1)-state recurrence — **no KV cache growth** | Cited; the reason the backbone is SSM-based |
| HippoRAG (2024) | Personalized-PageRank over a KG for long-term memory | Alternative to product keys for the relational store |
| MemGPT / Letta (2023) | OS-style paged memory tiers | Engineering pattern for the lifecycle manager |
| Larimar (IBM, 2024) | Episodic memory, one-shot knowledge edits | The "learn without retraining" claim, made concrete |
| MLX (Apple) | Unified-memory array framework; native 4-bit; LoRA/QLoRA | **The implementation stack.** See [04](04_architecture.md) |
| DeepSeek-V3 / Mixtral / GLaM | Production sparse-activation ratios | Baseline activation ratios to beat |

---

## What the literature map changes about the plan

1. **The trillion-parameter target is reachable — via product keys (A1), not via bigger MoE.**
   89 MB of index addressing 974B params is the difference between possible and not.
2. **Memory params are a proven substitute for compute on knowledge, not on reasoning (A2).**
   Set expectations and evaluation accordingly.
3. **Training the memory is impossible; writing it is not (A2 vs E2).** The whole program hinges here.
4. **Ternary + heavily-pretrained checkpoints is a known-bad interaction (C2).** Gate it early.
5. **c-BTM (D1) is the only single-machine-compatible training paradigm published.** Use it.
6. **Fine-grained experts are on the measured optimal frontier (D3), not a stylistic choice.**
7. **Titans (E1) supersedes the hand-written lifecycle heuristics.** Adopt, don't reinvent.
8. **SAEs (E3) make "pattern formation" measurable.** Without this, Component 2 is untestable.
