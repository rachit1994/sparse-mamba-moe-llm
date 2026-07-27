# 01 — Feasibility & Hardware Envelope

> All numbers reproduced by `verify_math.py` §4–§7, §12. Run it; don't trust this file.

---

## 1. What the machine actually is

Target: **Mac mini (2024), Apple M4, 16 GB unified memory.**

| Quantity | Value | Confidence |
|---|---|---|
| Unified memory | 16 GiB = 17,179,869,184 B = 17.18 GB decimal | Apple spec |
| Memory bandwidth, peak | 120 GB/s (LPDDR5X) | Apple spec |
| CPU | 10-core (4P + 6E) | Apple spec |
| GPU | 10-core, ~4.4 TFLOP/s FP32 | third-party benchmark |
| Neural Engine | 16-core, 38 TOPS INT8 | Apple spec |
| Internal SSD, sequential read | ~3 GB/s (256 GB model; ≥3 GB/s at 512 GB+) | third-party benchmark |
| Thunderbolt | TB4, ~40 Gbps ≈ 5 GB/s theoretical, ~3 GB/s real | spec (TB5 is M4 **Pro** only) |
| SSD random-read IOPS | **UNVERIFIED — must measure** | see `bench/` |

Two of these deserve emphasis because they are commonly misquoted:

- **120 GB/s, not 273 GB/s.** 273 GB/s is the M4 **Pro**. The base M4 in the 16 GB Mac mini has
  120 GB/s. Every throughput number in this research uses 120.
- **TB4, not TB5.** An external NVMe enclosure on this machine will not exceed ~3 GB/s. There is no
  configuration of this Mac mini where external storage is faster than the internal SSD.

## 2. Usable RAM, honestly accounted

```
  16.00 GiB   physical
-  3.50 GiB   macOS + WindowServer floor
-  1.00 GiB   Python/runtime/app overhead
-  1.50 GiB   headroom to avoid swap death
= 10.00 GiB   theoretical engine budget
```

macOS additionally caps GPU-wired memory at ~66–75% of physical RAM by default
(≈10.6–12.0 GiB on a 16 GB machine), adjustable via `sysctl iogpu.wired_limit_mb`. Raising it is
possible but reduces the OS's non-wirable working set; community guidance is to stay at or below
~70% of physical RAM.

**Design point adopted throughout: 9.0 GiB = 9.66 GB.** This leaves margin against both the OS
floor and the GPU wired cap. Anything that does not fit in 9.66 GB does not fit.

## 3. Can 10¹² parameters even be stored?

| Format | bytes/param | 10¹² params | Fits after macOS on 256 GB SSD? |
|---|---:|---:|---|
| FP16 | 2.0000 | 2000.0 GB | no |
| INT8 | 1.0000 | 1000.0 GB | no |
| INT4 | 0.5000 | 500.0 GB | no |
| 2-bit packed | 0.2500 | 250.0 GB | no |
| **ternary, 5 values per byte** | **0.2000** | **200.0 GB** | **marginal** |
| 1-bit | 0.1250 | 125.0 GB | yes |

Note on ternary packing: BitNet b1.58 is described as 1.58 bits/param because log₂3 = 1.585. In
practice you pack **5 ternary values into one byte** (3⁵ = 243 ≤ 256), giving **1.6 bits/param =
0.2 bytes/param**. Use 0.2, not 0.1975 — the difference is 1.3% of 200 GB.

**Storage verdict:** at ternary, 10¹² params = exactly 200 GB. The base 256 GB Mac mini has ~215 GB
free after macOS, so it fits with ~15 GB to spare and no room for anything else — including the
corpus, checkpoints, or the OS's own growth. **The 512 GB internal configuration is the minimum sane
build.** External NVMe over TB4 is an acceptable alternative at the same ~3 GB/s.

## 4. Decode roofline

Autoregressive decode is memory-bandwidth-bound, not compute-bound: each token reads every active
weight exactly once. So `tok/s ≤ effective_bandwidth ÷ (active_params × bytes_per_param)`.

At ternary (0.2 B/param), assuming 70–85% bandwidth efficiency (**assumption — measure it**):

| Active params | Bytes/token | tok/s @ 84 GB/s | tok/s @ 102 GB/s |
|---:|---:|---:|---:|
| 0.5 B | 0.10 GB | 840.0 | 1020.0 |
| 1.0 B | 0.20 GB | 420.0 | 510.0 |
| 2.0 B | 0.40 GB | 210.0 | 255.0 |
| 5.0 B | 1.00 GB | 84.0 | 102.0 |
| 10.0 B | 2.00 GB | 42.0 | 51.0 |
| 32.0 B | 6.40 GB | 13.1 | 15.9 |

This table is the whole design constraint. **To hit 30+ tok/s the active parameter set must be
under ~3B, and it must be resident.**

## 5. Why porting Kimi K2 does not work

Kimi K2 is a real, open-weights, 1-trillion-parameter model: 1T total, 32B active per token,
384 experts with 8 routed + 1 shared. It is the obvious thing to try, and it fails:

```
32B active × 0.2 B/param = 6.40 GB of weights per token
```

6.40 GB would fit the 9.66 GB budget *if it were the same 6.40 GB every token*. It is not — the
active set is re-selected per token from a 200 GB pool of which only ~6.5 GB (3.2%) can be resident.
Coarse-grained routing over 384 experts gives poor temporal reuse, so most of the 6.40 GB is fetched
from SSD each token:

| Expert-cache hit rate | s/token | tok/s |
|---:|---:|---:|
| 0% (cold) | 2.13 | 0.47 |
| 50% | 1.07 | 0.94 |
| 80% | 0.43 | 2.34 |
| 90% | 0.21 | 4.69 |

**Even at a 90% hit rate, K2 yields ~4.7 tok/s.** For reference, expert-offloading work
(Eliseev & Mazur 2023) achieves useful rates on Mixtral-8x7B — but Mixtral has 8 experts, not 384,
and ~47B total, not 1000B. The cache-to-pool ratio here is 30× worse.

**Conclusion: the activation set must be made both far smaller *and* far more reusable.** That is a
different architecture, not a deployment trick. This finding is what forces the design in
[04_architecture.md](04_architecture.md).

## 6. The binding constraint, stated precisely

```
time_per_token ≈ max( RAM_bytes / BW_ram , SSD_miss_bytes / BW_ssd , random_reads / IOPS )
```
with perfect prefetch overlap; the serialized sum if not.

SSD budget for a target rate, at 3 GB/s:

| Target | Bytes/token allowed to miss cache | ≈ ternary params |
|---:|---:|---:|
| 5 tok/s | 600.0 MB | 3000 M |
| 10 tok/s | 300.0 MB | 1500 M |
| 20 tok/s | 150.0 MB | 750 M |
| 30 tok/s | 100.0 MB | 500 M |
| 60 tok/s | 50.0 MB | 250 M |

With a 6.5 GB hot set (32.5B ternary params, 3.25% of the pool), required hit rates are:

| Active/token | @10 tok/s | @20 tok/s |
|---:|---:|---:|
| 1.0 B | none needed | 25.0% |
| 2.0 B | 25.0% | 62.5% |
| 5.0 B | 70.0% | 85.0% |
| 10.0 B | 85.0% | 92.5% |

**The design should aim to need *no* hit rate at all** — i.e. keep the entire routed-expert pool
resident and let only the memory layer touch SSD. That is what [04](04_architecture.md) does, and it
converts a fragile cache-hit-rate dependency into a hard guarantee.

## 7. Test-time compute is affordable; wall-clock is the cost

| Rate | 10³ thinking tokens | 10⁴ | 10⁵ |
|---:|---:|---:|---:|
| 10 tok/s | 100 s | 1000 s | 10000 s |
| 30 tok/s | 33 s | 333 s | 3333 s |
| 60 tok/s | 17 s | 167 s | 1667 s |
| 120 tok/s | 8 s | 83 s | 833 s |

A Mac mini at ~30 W for 120 s consumes ≈1.0 Wh ≈ $0.0002 of electricity. For comparison, the
ARC-AGI-2 compute-capped track budgets ~$0.20/task. **Energy is not the constraint by three orders of
magnitude.** Wall-clock latency and reasoning quality are the constraints. This is a strong argument
for spending throughput on search/reasoning depth rather than on more parameters.

---

## Feasibility verdict

| Question | Answer |
|---|---|
| Can 10¹² params be *stored* on this machine? | Yes, at ternary, at exactly 200 GB — needs the 512 GB model. |
| Can 10¹² params be *trained* on this machine? | **No.** Off by ~10⁶ (see [05](05_training_and_population.md)). |
| Can 10¹² params be *written* on this machine? | Yes — ~16 days for 974B of them. This is the unlock. |
| Can an existing 1T model (K2) be run? | Technically yes at 0.47–4.7 tok/s. Practically no. |
| Can a *purpose-built* 10¹²-param system run usefully? | Yes — est. ~37 tok/s. See [04](04_architecture.md). |
| Will it match a frontier dense model? | **Almost certainly not on reasoning.** See [06](06_evaluation.md). |

---

## Sources

- [Mac mini — Technical Specifications, Apple](https://www.apple.com/mac-mini/specs/)
- [Mac mini (2024) Tech Specs, Apple Support](https://support.apple.com/en-us/121555)
- [Apple M4 GPU (10-core) FP32 benchmark, cpu-monkey](https://www.cpu-monkey.com/en/igpu-apple_m4_10_core)
- [M4 Mac mini review — SSD speeds, AppleInsider](https://appleinsider.com/articles/24/11/13/m4-mac-mini-review-the-first-redesign-in-years-hides-incredible-computing-power)
- [Adjust VRAM/RAM split on Apple Silicon, llama.cpp discussion #2182](https://github.com/ggml-org/llama.cpp/discussions/2182)
- [Kimi K2 Technical Report, arXiv:2507.20534](https://arxiv.org/pdf/2507.20534)
- [Kimi K2 repository, MoonshotAI](https://github.com/moonshotai/kimi-k2)
- [BitNet b1.58 2B4T Technical Report, arXiv:2504.12285](https://arxiv.org/html/2504.12285v2)
- [Fast Inference of MoE Language Models with Offloading, arXiv:2312.17238](https://arxiv.org/abs/2312.17238)
- [ARC-AGI-2, ARC Prize](https://arcprize.org/arc-agi/2)
