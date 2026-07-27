#!/usr/bin/env python3
"""
Numbers backing the method choice in 09_method_comparison_and_decision.md.
Run: python3 verify_decision.py
"""
from math import sqrt

GB = 10**9
BPP = 0.2                 # ternary, 5 values/byte
BW_EFF = 84 * GB          # decode-effective RAM bandwidth (see 01)
SSD_SEQ = 3.0 * GB
IOPS = 300e3              # assumed; MUST be measured (bench/)

def hdr(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)

# ---------------------------------------------------------------------------
hdr("1. IO COST PER TOKEN — learned top-k gather vs hash lookup vs block fetch")

D_MODEL = 2048

print("A) Learned product keys, top-k gather (my original 04 spec):")
topk, mem_layers = 128, 8
reads_a = topk * mem_layers
print(f"   top-k={topk} x {mem_layers} layers        = {reads_a} random reads/token")
print(f"   at {IOPS/1e3:.0f}K IOPS                    = {1e3*reads_a/IOPS:.2f} ms/token"
      f" -> {IOPS/reads_a:.0f} tok/s ceiling")
print(f"   PREFETCHABLE? NO - the query is the residual stream, unknown until the layer runs.")

print("\nB) Meta-corrected: shared pool, 3 memory layers:")
reads_b = topk * 3
print(f"   top-k={topk} x 3 layers          = {reads_b} random reads/token")
print(f"   at {IOPS/1e3:.0f}K IOPS                    = {1e3*reads_b/IOPS:.2f} ms/token"
      f" -> {IOPS/reads_b:.0f} tok/s ceiling")
print(f"   PREFETCHABLE? NO - same reason.")

print("\nC) Engram-style n-gram hash (heads x n-gram orders x injection points):")
heads, orders, inject = 4, 2, 3
reads_c = heads * orders * inject
print(f"   {heads} heads x {orders} orders x {inject} inject pts = {reads_c} reads/token")
print(f"   at {IOPS/1e3:.0f}K IOPS                    = {1e3*reads_c/IOPS:.3f} ms/token"
      f" -> {IOPS/reads_c:,.0f} tok/s ceiling")
print(f"   PREFETCHABLE? YES - hash of token IDs, known before any layer runs.")
print(f"   reduction vs (A): {reads_a/reads_c:.0f}x fewer reads")

print("\nD) Apple-style block fetch (per CONTEXT, not per token):")
block_params = 18e6 * (2e9 / 160e6)      # scale their 18M block to a 2B backbone
print(f"   block = {block_params/1e6:.0f}M params = {block_params*BPP/1e6:.1f} MB"
      f" -> ONE sequential read of {block_params*BPP/1e6:.1f} MB")
print(f"   at {SSD_SEQ/GB:.0f} GB/s = {1e3*block_params*BPP/SSD_SEQ:.1f} ms per context switch")
print(f"   amortized over a 1000-token generation: "
      f"{1e3*block_params*BPP/SSD_SEQ/1000:.4f} ms/token")

# ---------------------------------------------------------------------------
hdr("2. THE PREFETCH ARGUMENT — why deterministic addressing wins on this box")

print("Decode is sequential: token t+1's n-gram key depends only on tokens <= t.")
print("So the memory read for step t+1 can be issued DURING step t's compute.\n")
t_compute = 6.19e-3      # ms->s, from 01: bandwidth-bound step time
t_io_c = reads_c / IOPS
t_io_b = reads_b / IOPS
print(f"   compute per token          : {1e3*t_compute:6.2f} ms")
print(f"   (C) hash IO per token      : {1e3*t_io_c:6.2f} ms")
print(f"   (B) top-k IO per token     : {1e3*t_io_b:6.2f} ms")
print(f"\n   NOT prefetchable (serialize): {1/(t_compute+t_io_b):6.1f} tok/s  [learned keys]")
print(f"   PREFETCHED (fully hidden)   : {1/max(t_compute,t_io_c):6.1f} tok/s  [hash keys]")
print(f"   speedup from prefetchability: {(t_compute+t_io_b)/max(t_compute,t_io_c):.2f}x")
print("\n   During PREFILL the entire sequence's keys are known at once ->")
print("   all memory reads can be issued as one batched, sorted, near-sequential pass.")

print("\n   SENSITIVITY TO IOPS (the one hardware number we have NOT measured):")
print(f"   {'IOPS':>10}{'learned-key tok/s':>20}{'hash tok/s':>13}{'hash advantage':>17}")
for io in (500e3, 300e3, 100e3, 50e3, 20e3):
    tb = reads_b / io
    tc = reads_c / io
    serial = 1 / (t_compute + tb)
    pref = 1 / max(t_compute, tc)
    print(f"   {io/1e3:>8.0f}K{serial:>20.1f}{pref:>13.1f}{pref/serial:>16.2f}x")
print("   => the hash design DEGRADES GRACEFULLY as IOPS falls; the learned-key")
print("      design does not. Choosing hash de-risks the untested assumption.")

# ---------------------------------------------------------------------------
hdr("3. HOW BIG SHOULD THE MEMORY ACTUALLY BE? (published ratios, scaled)")

BACKBONE = 2.0e9
print(f"Backbone assumed: {BACKBONE/1e9:.1f}B params\n")
print(f"{'source':<34}{'their ratio':>13}{'scaled memory':>16}{'disk @ternary':>15}")
refs = [
    ("Google Gemma 4 E4B (shipped)",      8.0/4.5,      "raw:effective"),
    ("Google Gemma 4 26B-A4B (shipped)",  26.0/3.8,     "total:activated"),
    ("Meta Memory Layers (max pub.)",     128.0/8.0,    "memory:base"),
    ("Apple hierarchical (edge-focused)", 4.6/0.160,    "bank:model"),
    ("ByteDance UltraMemV2 (max pub.)",   120.0/2.5,    "total:activated"),
]
for name, ratio, kind in refs:
    mem = BACKBONE * ratio
    print(f"{name:<34}{ratio:>12.1f}x{mem/1e9:>14.1f}B{mem*BPP/GB:>13.1f} GB")

print(f"\n{'THIS PROPOSAL (04_architecture)':<34}{974.0/26.0:>12.1f}x{974.0:>14.1f}B"
      f"{974e9*BPP/GB:>13.1f} GB")

max_pub = max(r for _, r, _ in refs)
print(f"\nMost aggressive PUBLISHED ratio : {max_pub:.1f}x  (UltraMemV2)")
print(f"This proposal's ratio           : {974.0/26.0:.1f}x")
print(f"Extrapolation beyond published  : {(974.0/26.0)/max_pub:.1f}x")
print(f"Largest memory bank ever published: 128B (Meta)")
print(f"This proposal                     : 974B  -> {974/128:.1f}x beyond")

# ---------------------------------------------------------------------------
hdr("4. STAGED TARGET — where to actually build, and what it costs")

print(f"{'stage':<10}{'memory params':>15}{'total':>10}{'disk':>11}{'vs published':>15}")
stages = [
    ("S1", 30e9),
    ("S2", 100e9),
    ("S3", 300e9),
    ("S4", 974e9),
]
for nm, mem in stages:
    tot = mem + 26e9
    print(f"{nm:<10}{mem/1e9:>13.0f}B{tot/1e9:>9.0f}B{tot*BPP/GB:>9.1f} GB"
          f"{mem/128e9:>13.1f}x")

print("\nStorage feasibility per stage (256 GB Mac mini has ~215 GB free):")
for nm, mem in stages:
    tot_disk = (mem + 26e9) * BPP / GB
    print(f"  {nm}: {tot_disk:6.1f} GB -> "
          f"{'fits 256GB model' if tot_disk < 200 else 'needs 512GB / external NVMe'}")

# ---------------------------------------------------------------------------
hdr("5. UltraMemV2's CONTROLLED RESULT — the one that should change the target")

print("Reported comparison at EQUAL activated compute (2.5B):")
print("    UltraMemV2-2.5B / 60B total  / top-768   >   UltraMemV2-2.5B / 120B total / top-256")
print("\n  i.e. 3x MORE activation density beat 2x MORE total sparse parameters.")
print("  Interpretation: at fixed compute, the marginal value of TOTAL sparse params")
print("  is lower than the marginal value of ACTIVATION DENSITY (top-m).\n")
print("  Consequence for this project: 'one trillion parameters' optimises the axis")
print("  with the WORST measured return. Budget should shift toward higher top-m")
print("  and a larger dense/active core, with total memory sized to the measured")
print("  point where the U-curve turns -- not fixed at 1e12 by fiat.")

# ---------------------------------------------------------------------------
hdr("6. REVISED BUDGET at stage S2 (100B memory) on the 16 GB box")

backbone, experts, memory = 2.0e9, 24.0e9, 100e9
resident = (backbone + experts) * BPP
ram_per_tok = (backbone + 0.6e9) * BPP
print(f"  backbone {backbone/1e9:.0f}B + experts {experts/1e9:.0f}B resident"
      f" = {resident/GB:.2f} GB")
print(f"  memory {memory/1e9:.0f}B on SSD              = {memory*BPP/GB:.1f} GB")
print(f"  TOTAL system                       = {(backbone+experts+memory)/1e9:.0f}B params,"
      f" {(backbone+experts+memory)*BPP/GB:.1f} GB disk")
print(f"  RAM: {resident/GB:.2f} weights + 0.54 KV + 0.50 act = "
      f"{resident/GB + 0.54 + 0.50:.2f} GB of 9.66 GB budget")
print(f"  decode bandwidth limit             = {BW_EFF/ram_per_tok:.0f} tok/s")
print(f"  hash-memory IO (prefetched)        = hidden under compute")
print(f"  => S2 fits the 256 GB base Mac mini with room to spare.")
