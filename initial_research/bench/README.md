# bench/ — Measurements That Must Be Taken on the Actual Mac mini

Everything in [`../01_feasibility.md`](../01_feasibility.md) rests on six numbers. Four are vendor
specs or third-party benchmarks; **two have never been measured on this machine** and one of those
(random-read IOPS) originally drove an architecture decision.

**These are gate G0 in [`../07_kill_switches.md`](../07_kill_switches.md). Run them before writing
any model code.**

> ⚠️ **None of these have been run.** This research was conducted in a Linux container, not on the
> target hardware. Every throughput figure in `initial_research/` is therefore a *projection from
> vendor specs plus stated efficiency assumptions*, not a measurement. Treat them as such until this
> directory has output in it.

---

## What to measure

| ID | Quantity | Assumed | Why it matters |
|---|---|---|---|
| G0.1 | Decode-effective memory bandwidth | 84–102 GB/s (70–85% of 120) | Sets the tok/s ceiling — **the binding constraint** |
| G0.2 | Sustained GPU MFU (MLX) | 30% → 1.32 TFLOP/s | Sets every training and encode timeline in [05](../05_training_and_population.md) |
| G0.3 | SSD sequential read | ~3 GB/s | Tier-B block fetch |
| G0.4 | SSD random 4 KB read IOPS | 300K (**never measured**) | Tier-A; design was chosen to be insensitive to this |
| G0.5 | Max stable `iogpu.wired_limit_mb` | ~12 GiB | Expert pool size |
| G0.6 | Ternary kernel bandwidth efficiency | ≥50% of fp16 | Whether 0.2 B/param is real in practice |

---

## G0.1 — Memory bandwidth

The number that matters is **decode-shaped** bandwidth: a large GEMV streaming weights once.
`STREAM`-style triad benchmarks overstate it.

```bash
# Practical proxy: run a known memory-bound decode and back out bandwidth.
# A 7B Q4_K_M model reads ~4.1 GB/token.
brew install llama.cpp     # or build from source
llama-bench -m <7b-q4_k_m.gguf> -p 0 -n 128 -r 3
# effective_bandwidth ≈ tok/s × bytes_per_token
#   e.g. 30 tok/s × 4.1 GB = 123 GB/s  → implausible, re-check the file size
#        25 tok/s × 4.1 GB = 102 GB/s  → 85% efficiency
```

Record: model file size (bytes), measured tok/s, derived GB/s, derived % of 120 GB/s peak.

**Pass: ≥ 84 GB/s.** Below that, halve every tok/s figure in [01](../01_feasibility.md) proportionally.

## G0.2 — Sustained GPU MFU

```bash
pip install mlx
python - <<'PY'
import mlx.core as mx, time
N = 4096
a = mx.random.normal((N, N), dtype=mx.float16)
b = mx.random.normal((N, N), dtype=mx.float16)
mx.eval(a, b)
for _ in range(3): mx.eval(a @ b)          # warm up
t0 = time.perf_counter(); iters = 50
for _ in range(iters): c = a @ b
mx.eval(c)
dt = time.perf_counter() - t0
flops = 2 * N**3 * iters
print(f"{flops/dt/1e12:.2f} TFLOP/s  ({100*flops/dt/4.4e12:.0f}% of 4.4 TFLOP/s FP32 peak)")
PY
```

Run for **≥10 minutes** to capture thermal steady state — a Mac mini throttles, and the sustained
number is the one that sets the encode timeline in [05](../05_training_and_population.md).

**Pass: ≥ 1.32 TFLOP/s sustained.**

## G0.3 / G0.4 — SSD sequential and random read

Sequential:
```bash
# Write a file larger than RAM so the page cache cannot serve it (16 GiB RAM → use 32 GB).
mkfile -n 32g /tmp/bench.bin        # sparse; or dd for a real file
sudo purge                          # drop caches — REQUIRED or you measure RAM
dd if=/tmp/bench.bin of=/dev/null bs=1m count=8192
```

Random 4 KB — **the unmeasured number**:
```bash
brew install fio
sudo purge
fio --name=rand4k --filename=/tmp/bench.bin --rw=randread \
    --bs=4k --iodepth=32 --numjobs=4 --direct=1 \
    --runtime=60 --time_based --group_reporting
```

Record IOPS at queue depths 1, 8, 32, 128. Depth-1 is the worst case for a non-prefetched design;
high depth is what a *prefetched* design achieves — that gap is the value of the choice made in
[09 §3](../09_method_comparison_and_decision.md).

**Pass: ≥ 50K IOPS at depth 32.** The chosen architecture holds 161.6 tok/s down to 20K IOPS, so
this gate is lenient by design.

## G0.5 — GPU wired memory limit

```bash
sysctl iogpu.wired_limit_mb                 # current
sudo sysctl iogpu.wired_limit_mb=12288      # try 12 GiB
# then run a 1-hour sustained load and watch for swap:
vm_stat 60 | awk '{print $0}'               # watch "Swapouts"
```

**Pass: no sustained swapouts over 1 hour.** Reset with `sudo sysctl iogpu.wired_limit_mb=0`.
Do not exceed ~70% of physical RAM; macOS needs non-wirable memory.

## G0.6 — Ternary kernel efficiency

```bash
git clone --recursive https://github.com/microsoft/BitNet && cd BitNet
# follow repo setup, then run their bench against bitnet-b1.58-2B-4T
# compare achieved GB/s to the fp16 baseline from G0.1
```

**Pass: ≥ 50% of fp16 bandwidth efficiency.** Below that, ternary's 0.2 B/param storage win does not
translate into a throughput win, and INT4 becomes the better trade.

---

## Reporting

Write results to `results_<YYYY-MM-DD>.md` in this directory with, for each ID: the exact command,
raw output, derived number, and **pass/fail against the threshold**. Then update
[`../01_feasibility.md`](../01_feasibility.md) — replacing assumptions with measurements — and re-run
both verification scripts so every downstream number reflects reality:

```bash
python3 ../verify_math.py     > ../verify_math_output.txt
python3 ../verify_decision.py > ../verify_decision_output.txt
```

If any G0 threshold fails, **do not proceed to G1** until the affected numbers have been re-derived.
Patching a threshold to make it pass is the specific failure this directory exists to prevent.
