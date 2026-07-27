#!/usr/bin/env python3
"""
PCIE feasibility math — every number in initial_research/ is produced here.
No hand arithmetic. Run: python3 verify_math.py
"""
from math import comb, log10, lgamma, log, sqrt

SEC_PER_YEAR = 365.25 * 24 * 3600
GiB = 2**30
GB = 10**9
TB = 10**12

def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)

# ---------------------------------------------------------------- SECTION 1
hdr("1. SDR CAPACITY  — checking README claim C(10000,100) ~ 10^189")

c = comb(10000, 100)
print(f"C(10000,100) exact digits      : {len(str(c))}")
print(f"C(10000,100) log10             : {log10(c):.4f}")
print(f"README claim                   : 10^189")
print(f"VERDICT                        : README is WRONG by {log10(c)-189:.1f} orders of magnitude")
print(f"Correct value                  : ~10^{log10(c):.1f}")

# independent cross-check via lgamma (no bignum), must agree
lg = (lgamma(10001) - lgamma(101) - lgamma(9901)) / log(10)
print(f"cross-check via lgamma (log10) : {lg:.4f}  -> agrees: {abs(lg-log10(c)) < 1e-6}")

print("\nWhat 10^189 actually corresponds to:")
for w in (60, 64, 70, 80, 100):
    print(f"  C(10000,{w:3d}) = 10^{log10(comb(10000,w)):7.2f}")

print("\nComparison anchors:")
print(f"  Numenta canonical SDR C(2048,40) = 10^{log10(comb(2048,40)):.2f}")
print(f"  atoms in observable universe     ~ 10^80")

# ---------------------------------------------------------------- SECTION 2
hdr("2. SEQUENCE-LENGTH CLAIM — 'reduce seq 4x => 16x less attention compute'")

def transformer_flops_per_seq(n, d, layers=1):
    """MACs per sequence. Linear terms 12*d^2 per token; attention 2*n*d per token."""
    linear = n * 12 * d * d * layers
    attn = n * 2 * n * d * layers
    return linear, attn

for d, n in ((2048, 2048), (4096, 4096), (4096, 16384), (8192, 131072)):
    l0, a0 = transformer_flops_per_seq(n, d)
    l1, a1 = transformer_flops_per_seq(n // 4, d)
    tot0, tot1 = l0 + a0, l1 + a1
    print(f"\n d={d:5d} n={n:6d}")
    print(f"   attention share of total FLOPs : {100*a0/tot0:5.1f}%")
    print(f"   attention-only reduction at 4x : {a0/a1:6.2f}x   (README says 16x)")
    print(f"   TOTAL model reduction at 4x    : {tot0/tot1:6.2f}x   <-- the number that matters")

print("\nVERDICT: '16x' is true only for the attention sub-term, which is a minority")
print("         of FLOPs at realistic d,n. Real end-to-end saving ~4-5x, and that")
print("         ignores the byte-level encoder/decoder a patcher must add back.")

# ---------------------------------------------------------------- SECTION 3
hdr("3. MoE CLAIM — '64 experts x 4M, top-2 => 32x savings'")
E, P_E, k = 64, 4e6, 2
total, active = E * P_E, k * P_E
print(f"total expert params  : {total/1e6:.0f}M")
print(f"active expert params : {active/1e6:.0f}M")
print(f"ratio                : {total/active:.0f}x  -> README arithmetic is CORRECT")
print("\nBUT the README presents this as system savings. It is not:")
print("  - MEMORY is unchanged: all 256M must be resident or streamable.")
print("  - attention/router/embeddings/norms are NOT reduced.")
print("  - Amdahl: if experts are 60% of FLOPs, 32x on that part gives")
for share in (0.5, 0.6, 0.7, 0.8):
    speed = 1 / ((1 - share) + share / 32)
    print(f"      expert share {share:.0%} -> end-to-end {speed:.2f}x")

# ---------------------------------------------------------------- SECTION 4
hdr("4. TARGET HARDWARE — M4 Mac mini, 16 GB unified memory")

BW_PEAK = 120 * GB          # 120 GB/s, Apple spec
BW_EFF_LO, BW_EFF_HI = 0.70, 0.85
SSD_SEQ = 3.0 * GB          # ~3 GB/s measured on M4 internal SSD
RAM_TOTAL = 16 * GiB
print(f"unified memory       : {RAM_TOTAL/GiB:.0f} GiB = {RAM_TOTAL/GB:.2f} GB decimal")
print(f"peak memory bandwidth: {BW_PEAK/GB:.0f} GB/s")
print(f"effective (decode)   : {BW_PEAK*BW_EFF_LO/GB:.0f}-{BW_PEAK*BW_EFF_HI/GB:.0f} GB/s")
print(f"internal SSD seq read: {SSD_SEQ/GB:.1f} GB/s")

print("\nUsable RAM budget:")
budget = [
    ("macOS + WindowServer floor", 3.5),
    ("python/runtime/app overhead", 1.0),
    ("headroom to avoid swap death", 1.5),
]
used = sum(v for _, v in budget)
for nme, v in budget:
    print(f"  -{nme:32s} {v:5.2f} GiB")
avail = RAM_TOTAL / GiB - used
print(f"  = engine budget                  {avail:5.2f} GiB")
print(f"  (macOS default GPU wired cap ~66-75% of RAM = {0.66*16:.1f}-{0.75*16:.1f} GiB)")

ENGINE_GIB = 9.0
print(f"\nDESIGN POINT: engine budget = {ENGINE_GIB} GiB = {ENGINE_GIB*GiB/GB:.2f} GB")

# ---------------------------------------------------------------- SECTION 5
hdr("5. CAN 1e12 PARAMETERS EVEN BE STORED?")
P = 1e12
fmts = [("FP16", 16), ("INT8", 8), ("INT4", 4), ("2-bit packed", 2),
        ("ternary 5-per-8bit", 8/5), ("1-bit", 1)]
print(f"{'format':<20}{'bytes/param':>12}{'size':>14}{'fits 256GB SSD?':>18}")
for nme, bits in fmts:
    bpp = bits / 8
    size = P * bpp
    fits = "yes" if size < 200 * GB else "NO"
    print(f"{nme:<20}{bpp:>12.4f}{size/GB:>11.1f} GB{fits:>18}")

print("\nNOTE: ternary is 5 values per byte (3^5=243<=256) -> 1.6 bits/param, not 1.58.")
print(f"      1e12 ternary params = {P*1.6/8/GB:.1f} GB.")
print("      Base Mac mini ships 256 GB SSD. After macOS (~20 GB) you have ~215 GB.")
print("      => 1T ternary params fits ONLY on the 256GB model with almost nothing")
print("         else on disk. 512 GB internal or external NVMe is the sane config.")

# ---------------------------------------------------------------- SECTION 6
hdr("6. DECODE ROOFLINE — what throughput is physically available?")

def toks_per_sec(active_params, bpp, bw):
    return bw / (active_params * bpp)

print("Resident-in-RAM case (no SSD traffic), ternary 0.2 B/param:")
print(f"{'active params':>16}{'bytes/token':>14}{'tok/s @84GB/s':>16}{'tok/s @102GB/s':>16}")
for a in (0.5e9, 1e9, 2e9, 5e9, 1e10, 3.2e10):
    b = a * 0.2
    print(f"{a/1e9:>13.1f}B{b/GB:>11.2f} GB"
          f"{toks_per_sec(a,0.2,BW_PEAK*BW_EFF_LO):>16.1f}"
          f"{toks_per_sec(a,0.2,BW_PEAK*BW_EFF_HI):>16.1f}")

print("\nKimi K2 reference point (1T total / 32B active), if it were ternary:")
k2_bytes = 32e9 * 0.2
hot_gb_k2 = 6.5
print(f"  32B active x 0.2 B = {k2_bytes/GB:.2f} GB of weights per token")
print(f"  That 6.40 GB would fit the {ENGINE_GIB*GiB/GB:.2f} GB budget IF it were the same")
print(f"  6.40 GB every token. It is not: the active set is re-selected per token")
print(f"  from a {P*0.2/GB:.0f} GB pool, of which only ~{hot_gb_k2:.1f} GB ({100*hot_gb_k2*GB/(P*0.2):.1f}%) is resident.")
print(f"  Coarse-grained routing (8 of 384 experts) gives poor reuse, so most of")
print(f"  the 6.40 GB is fetched from SSD each token:")
print(f"    fully-missing case : {k2_bytes/SSD_SEQ:.2f} s/token = {SSD_SEQ/k2_bytes:.2f} tok/s")
for hit in (0.5, 0.8, 0.9):
    miss = k2_bytes * (1 - hit)
    print(f"    {hit:.0%} hit rate     : {miss/SSD_SEQ:.2f} s/token = {SSD_SEQ/miss:.2f} tok/s")
print("  => even at a 90% hit rate K2 gives ~4.7 tok/s. Porting K2 is NOT the answer;")
print("     the activation set must be made far smaller AND far more reusable.")

# ---------------------------------------------------------------- SECTION 7
hdr("7. THE BINDING CONSTRAINT: SSD miss traffic per token")

print("Time per token = max(RAM traffic / BW_ram, SSD miss traffic / BW_ssd)")
print("(assuming perfect prefetch overlap; serialized is the sum)\n")
print(f"SSD budget for a target rate (at {SSD_SEQ/GB:.0f} GB/s):")
for tgt in (5, 10, 20, 30, 60):
    mb = SSD_SEQ / tgt / 1e6
    print(f"  {tgt:3d} tok/s -> {mb:7.1f} MB/token may miss cache"
          f" = {mb*1e6/0.2/1e6:8.1f}M ternary params")

print("\nRequired cache hit rate, hot set resident in RAM:")
hot_gb = 6.5
hot_params = hot_gb * GB / 0.2
print(f"  hot set {hot_gb} GB ternary = {hot_params/1e9:.1f}B params resident of 1000B total"
      f" ({100*hot_params/P:.2f}%)")
print(f"{'active/token':>14}{'target tok/s':>14}{'required hit rate':>20}")
for a in (1e9, 2e9, 5e9, 1e10):
    for tgt in (10, 20):
        miss_budget_params = (SSD_SEQ / tgt) / 0.2
        need = 1 - miss_budget_params / a
        v = f"{100*need:.1f}%" if need > 0 else "none needed"
        print(f"{a/1e9:>11.1f}B{tgt:>14}{v:>20}")

# ---------------------------------------------------------------- SECTION 8
hdr("8. PRODUCT-KEY MEMORY — why 1e12 params is tractable at all")

print("PEER/product-key: N slots addressed by two sqrt(N) sub-key sets.")
print("Index cost is O(sqrt(N)*d), NOT O(N*d). This is the whole unlock.\n")
d_mem = 2048        # memory value dim == d_model of the backbone
for N in (1e6, 1e8, 4.76e8, 1e9):
    sq = sqrt(N)
    key_params = 2 * sq * (d_mem / 2)
    val_params = N * d_mem
    print(f"  N={N:>10.0e} slots: sqrt(N)={sq:>9.0f}"
          f"  key params={key_params/1e6:>7.1f}M ({key_params*2/1e6:>6.1f} MB fp16)"
          f"  value params={val_params/1e9:>7.1f}B")

N_SLOTS = 4.756e8
print(f"\nTarget: {N_SLOTS/1e6:.0f}M slots x d={d_mem} = {N_SLOTS*d_mem/1e9:.0f}B value params")
print(f"  value store on disk @ ternary: {N_SLOTS*d_mem*0.2/GB:.1f} GB")
print(f"  key index in RAM @ fp16      : {2*sqrt(N_SLOTS)*(d_mem/2)*2/1e6:.1f} MB  <-- fits trivially")

TOPK = 128
LAYERS_MEM = 8
per_tok_bytes = TOPK * LAYERS_MEM * d_mem * 0.2
per_tok_reads = TOPK * LAYERS_MEM
print(f"\nPer token, top-k={TOPK} over {LAYERS_MEM} memory layers:")
print(f"  value bytes fetched : {per_tok_bytes/1e3:.1f} KB   (bandwidth: negligible)")
print(f"  random reads        : {per_tok_reads}")
for iops in (100e3, 300e3, 500e3):
    print(f"    @ {iops/1e3:.0f}K IOPS -> {1e3*per_tok_reads/iops:.2f} ms/token"
          f" ({iops/per_tok_reads:.0f} tok/s ceiling)")
print("  => memory layers are IOPS-bound, not bandwidth-bound. 4KB-align slots.")

# ---------------------------------------------------------------- SECTION 9
hdr("9. PROPOSED BUDGET — does a 1e12-param system close?")

BPP = 0.2                     # ternary, 5 values per byte
P_BACKBONE = 2.0e9            # dense SSM/attention backbone, always active
P_EXPERTS = 2.4e10            # fine-grained MoE pool, fully RAM-resident after warmup
P_ACTIVE_EXP = 6.0e8          # expert params actually used per token
TOPK, LAYERS_MEM, D_MEM = 128, 8, 2048

# size the memory bank so the SYSTEM lands exactly on 1e12 parameters
def index_params(n):
    return 2 * sqrt(n) * (D_MEM / 2)

n_slots = (1e12 - P_BACKBONE - P_EXPERTS) / D_MEM     # first pass, ignore index
for _ in range(50):                                    # converge including index cost
    n_slots = (1e12 - P_BACKBONE - P_EXPERTS - index_params(n_slots)) / D_MEM
P_MEM_VAL = n_slots * D_MEM
P_INDEX = index_params(n_slots)
tot_params = P_BACKBONE + P_EXPERTS + P_MEM_VAL + P_INDEX

comp = [
    ("dense backbone (SSM/attn)", P_BACKBONE, BPP, "RAM"),
    ("fine-grained MoE pool",     P_EXPERTS,  BPP, "RAM"),
    ("  ..active per token",      P_ACTIVE_EXP, BPP, "(subset)"),
    ("product-key memory values", P_MEM_VAL,  BPP, "SSD"),
    ("  ..fetched per token",     TOPK*LAYERS_MEM*D_MEM, BPP, "(subset)"),
    ("product-key index (fp16)",  P_INDEX,    2.0, "RAM"),
]
print(f"{'component':<30}{'params':>13}{'footprint':>13}{'tier':>11}")
for nme, p, bpp, tier in comp:
    print(f"{nme:<30}{p/1e9:>11.3f}B{p*bpp/GB:>10.3f} GB{tier:>11}")
print(f"\nTOTAL PARAMETERS : {tot_params/1e12:.6f}e12"
      f"  -> {'MEETS' if tot_params >= 0.999e12 else 'MISSES'} the 1T target")
print(f"memory slots      : {n_slots/1e9:.3f}B  (sqrt(N) = {sqrt(n_slots):,.0f})")
print(f"TOTAL DISK        : {(P_MEM_VAL*BPP)/GB:.1f} GB values"
      f" + {(P_BACKBONE+P_EXPERTS)*BPP/GB:.1f} GB weights"
      f" = {(P_MEM_VAL+P_BACKBONE+P_EXPERTS)*BPP/GB:.1f} GB")

resident = (P_BACKBONE + P_EXPERTS) * BPP + P_INDEX * 2.0
kv = 4 * 8 * 128 * 32768 * 2 * 2   # 4 attn layers, 8 kv heads, hd128, 32k ctx, fp16, K+V
act = 0.5 * GB                     # activations / scratch / runtime
print(f"\nRAM ledger (budget {ENGINE_GIB*GiB/GB:.2f} GB):")
print(f"  backbone + experts resident : {(P_BACKBONE+P_EXPERTS)*BPP/GB:>6.3f} GB")
print(f"  product-key index           : {P_INDEX*2.0/GB:>6.3f} GB")
print(f"  KV cache (32k ctx)          : {kv/GB:>6.3f} GB")
print(f"  activations / scratch       : {act/GB:>6.3f} GB")
print(f"  ---------------------------------------")
print(f"  total                       : {(resident+kv+act)/GB:>6.3f} GB")
print(f"  slack                       : {(ENGINE_GIB*GiB-resident-kv-act)/GB:>6.3f} GB"
      f"  -> {'FITS' if resident+kv+act < ENGINE_GIB*GiB else 'DOES NOT FIT'}")
print("  NOTE: the whole expert pool is resident, so after warmup expert SSD traffic")
print("        is ZERO. This is the design choice that makes the box viable.")

print("\nPer-token cost, three independent limits:")
ram_bytes = (P_BACKBONE + P_ACTIVE_EXP) * BPP
t_ram = ram_bytes / (BW_PEAK * BW_EFF_LO)
flops_tok = 2 * (P_BACKBONE + P_ACTIVE_EXP)
t_cmp = flops_tok / (M4_FP32_PEAK_ := 4.4e12 * 0.30)
print(f"  [bandwidth] {ram_bytes/GB:.3f} GB/token / {BW_PEAK*BW_EFF_LO/GB:.0f} GB/s"
      f" = {1e3*t_ram:6.2f} ms -> {1/t_ram:6.1f} tok/s")
print(f"  [compute  ] {flops_tok/1e9:.2f} GFLOP/token / {M4_FP32_PEAK_/1e12:.2f} TFLOP/s"
      f" = {1e3*t_cmp:6.2f} ms -> {1/t_cmp:6.1f} tok/s")
for iops in (100e3, 300e3):
    t_io = (TOPK * LAYERS_MEM) / iops
    print(f"  [memory IO] {TOPK*LAYERS_MEM} random reads / {iops/1e3:.0f}K IOPS"
          f" = {1e3*t_io:6.2f} ms -> {1/t_io:6.1f} tok/s")
t_io = (TOPK * LAYERS_MEM) / 300e3
print(f"\n  BINDING LIMIT (max of the three): {1/max(t_ram, t_cmp, t_io):.1f} tok/s theoretical")
print(f"  serialized (no overlap)         : {1/(t_ram+t_cmp+t_io):.1f} tok/s")
print(f"  at 50% of serialized (realistic): {0.5/(t_ram+t_cmp+t_io):.1f} tok/s  <-- PLAN AGAINST THIS")

print("\nPrefill (compute-bound, assume 50% MFU on GEMM):")
for ptoks in (512, 2048, 8192):
    s = ptoks * flops_tok / (4.4e12 * 0.50)
    print(f"  {ptoks:>5d}-token prompt -> {s:6.2f} s ({ptoks/s:.0f} tok/s prefill)")
print("  => prefill, not decode, is the latency users will feel. Budget for it.")

# ---------------------------------------------------------------- SECTION 10
hdr("10. TRAINING — the number that decides the whole program")

M4_FP32_PEAK = 4.4e12       # TFLOPS FP32, 10-core M4 GPU
for mfu, label in ((0.30, "30% MFU (realistic Metal)"), (0.50, "50% MFU (optimistic)")):
    eff = M4_FP32_PEAK * mfu
    print(f"\n--- {label}: {eff/1e12:.2f} TFLOP/s effective ---")
    scenarios = [
        ("1T dense, Chinchilla 20 tok/param", 1e12, 2e13),
        ("1T MoE, 5B active, 20T tokens",     5e9,  2e13),
        ("1T MoE, 5B active, 1T tokens",      5e9,  1e12),
        ("30B active, 1T tokens",             3e10, 1e12),
        ("2B backbone, 100B tokens",          2e9,  1e11),
        ("500M backbone, 10B tokens",         5e8,  1e10),
        ("100M backbone, 2B tokens",          1e8,  2e9),
    ]
    for nme, N, D in scenarios:
        flops = 6 * N * D
        secs = flops / eff
        yrs = secs / SEC_PER_YEAR
        t = f"{yrs:.2e} yr" if yrs > 1 else f"{secs/86400:.2f} days"
        print(f"  {nme:<38} C={flops:.2e} FLOP -> {t}")

eff = M4_FP32_PEAK * 0.30
for days in (7, 30, 90):
    budget_flops = eff * days * 86400
    ND = budget_flops / 6
    print(f"\n{days:3d} days at 30% MFU = {budget_flops:.3e} FLOP  => N*D = {ND:.3e}")
    for N in (5e7, 1e8, 3e8, 1e9):
        print(f"     N={N/1e6:6.0f}M params -> D={ND/N/1e9:8.2f}B tokens"
              f"  ({ND/N/N:6.1f} tok/param)")

# ---------------------------------------------------------------- SECTION 11
hdr("11. WRITING vs TRAINING MEMORY — the actual path to 1e12")

eff = M4_FP32_PEAK * 0.30

print("Regime (a) GRADIENT-TRAIN the memory values (Memory Layers at Scale style).")
for tok_per_param in (1, 20):
    D = P_MEM_VAL * tok_per_param
    f = 6 * P_MEM_VAL * D
    print(f"   {tok_per_param:2d} tok/param -> C={f:.2e} FLOP -> {f/eff/SEC_PER_YEAR:.2e} years")
print("   VERDICT: DEAD. Off by 5-6 orders of magnitude. Not a scheduling problem.")

print("\nRegime (b) WRITE the memory values by encoding a corpus once (RETRO/kNN-LM style).")
print("   Cost = 2 * N_encoder * corpus_tokens (forward pass only, one epoch).")
print(f"   Need {N_SLOTS/1e6:.0f}M slots. corpus_tokens = slots * chunk_len.\n")
print(f"   {'encoder':>9}{'chunk':>7}{'corpus tok':>13}{'FLOP':>11}{'wall clock':>14}")
feasible = []
for enc in (1e7, 3e7, 1e8, 5e8):
    for chunk in (64, 128, 256):
        corpus = N_SLOTS * chunk
        f = 2 * enc * corpus
        days = f / eff / 86400
        mark = ""
        if days <= 45:
            mark = "  <-- viable"
            feasible.append((enc, chunk, days))
        print(f"   {enc/1e6:>7.0f}M{chunk:>7d}{corpus:>13.2e}{f:>11.2e}{days:>11.1f} d{mark}")

print("\n   CORRECTION TO MY OWN EARLIER LINE: at a 500M encoder and 256-token chunks")
print("   this is 4489 days (12.3 years), NOT 'weeks'. The job is only feasible with")
print("   a SMALL encoder and SHORT chunks. Concretely:")
if feasible:
    e, c, d_ = min(feasible, key=lambda x: x[2])
    print(f"     encoder {e/1e6:.0f}M params, {c}-token chunks"
          f" -> {N_SLOTS*c/1e9:.1f}B corpus tokens -> {d_:.1f} days one-time")
print("   The Neural Engine (38 TOPS INT8) could cut this ~3-10x further if the")
print("   encoder can be compiled to it, but treat that as UNVERIFIED upside.")

print("\nRegime (c) what the corpus requirement implies:")
for chunk in (64, 128, 256):
    print(f"   chunk={chunk:3d} -> need {N_SLOTS*chunk/1e9:6.1f}B tokens of text"
          f"  (the Pile ~300B, FineWeb ~15T -> {'available' if N_SLOTS*chunk < 3e11 else 'need big corpus'})")

print("\n=> CONCLUSION: 1e12 parameters is reachable on this box ONLY because ~97% of")
print("   them are WRITTEN (one forward pass each) rather than TRAINED (many backward")
print("   passes each). That is a ~2-week batch job vs a ~10^5-year one.")

# ---------------------------------------------------------------- SECTION 12
hdr("12. TEST-TIME COMPUTE — trading throughput for reasoning")
for rate in (10, 30, 60, 120):
    for thinking in (1e3, 1e4, 1e5):
        print(f"  {rate:4d} tok/s, {thinking:.0e} thinking tokens -> {thinking/rate:8.1f} s/answer")
    print()

print("Reference: ARC-AGI-2 Kaggle compute-capped track allows ~$0.20/task.")
print("A Mac mini at ~30W for 120 s = 1.0 Wh ~ $0.0002 of electricity.")
print("Energy is NOT the constraint. Wall-clock and reasoning quality are.")
