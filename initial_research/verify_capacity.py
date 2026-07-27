#!/usr/bin/env python3
"""
Information-capacity math for the fixed-substrate (dynamics-first) architecture.
This is the script that reversed the architecture decision in 09/10.
Run: python3 verify_capacity.py
"""
from math import comb, log2, log10

GB = 10**9
GiB = 2**30

def hdr(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)

# ---------------------------------------------------------------------------
hdr("1. PATTERN COUNT vs INFORMATION — the unit error in the root README")

c = comb(10000, 100)
print(f"C(10000,100)            = 10^{log10(c):.1f}   (count of distinguishable patterns)")
print(f"log2(C(10000,100))      = {log2(c):.1f} bits  (payload ONE pattern can carry)")
print()
print("A pattern COUNT is an address space. Its LOG is the storage capacity.")
print("10^241 patterns is not 10^241 bits of knowledge. It is 803 bits.")
print()
print("Upper bound regardless of encoding: an n-unit substrate holds <= n bits.")
for n in (10_000, 100_000, 1_000_000):
    k = n // 100
    print(f"   n={n:>9,} units, w={k:>6,} active:"
          f"  log2(C(n,w)) = {log2(comb(n,k)):>10,.0f} bits"
          f"  | raw ceiling {n:>9,} bits")
print("\nThe SDR argument shows ADDRESSING is not the bottleneck. True, and never")
print("was in dispute. It says nothing about how many BITS can be stored.")

# ---------------------------------------------------------------------------
hdr("2. KNOWLEDGE CAPACITY OF A FIXED BRAIN (Allen-Zhu & Li: 2 bits/param)")

RAM_WEIGHTS = 6.5 * GB          # weight budget inside the 9.66 GB engine budget
print(f"Weight budget in RAM: {RAM_WEIGHTS/GB:.1f} GB = {RAM_WEIGHTS*8/GB:.0f} Gbit")
print("  ^ this is a HARD information ceiling: you cannot store more bits than you have.\n")
print(f"{'precision':<10}{'bytes/param':>13}{'params':>12}{'2b/param':>12}{'actual (capped)':>18}")
for name, bpp in (("fp16", 2.0), ("int8", 1.0), ("int4", 0.5), ("ternary", 0.2)):
    p = RAM_WEIGHTS / bpp
    raw = 2 * p                       # Allen-Zhu 2 bits/param
    capped = min(raw, RAM_WEIGHTS * 8)  # cannot exceed physical storage bits
    flag = "  <- storage-bound" if raw > RAM_WEIGHTS * 8 else ""
    print(f"{name:<10}{bpp:>13.2f}{p/1e9:>10.1f}B{raw/1e9:>10.1f} Gb"
          f"{capped/1e9:>13.1f} Gb{flag}")
print("\nNOTE: Allen-Zhu verified 2 bits/param down to int8. Below int8 the law is")
print("      NOT verified, and the storage bound binds first. Treat int4/ternary")
print("      rows as UPPER bounds, not measurements. This is gate G-CAP.")

# ---------------------------------------------------------------------------
hdr("3. IS THAT ENOUGH? — what the ceiling actually buys")

CEIL = RAM_WEIGHTS * 8
targets = [
    ("Allen-Zhu reference: 7B model", 7e9 * 2,
     "they estimate this EXCEEDS English Wikipedia + textbooks combined"),
    ("English Wikipedia + textbooks", 14e9, "per the above"),
    ("30B-token web corpus (long tail)", 30e9 * 4 * 2, "~4 chars/token, ~2 bits/char compressed"),
    ("300B-token corpus (the Pile)", 300e9 * 4 * 2, ""),
]
print(f"Fixed-brain ceiling in {RAM_WEIGHTS/GB:.1f} GB: {CEIL/1e9:.0f} Gbit\n")
for name, bits, note in targets:
    verdict = "FITS" if CEIL >= bits else f"SHORT by {bits/CEIL:.1f}x"
    print(f"  {name:<36}{bits/1e9:>8.1f} Gb   {verdict}")
    if note:
        print(f"      {note}")

print("\n=> CONCLUSION THAT REVERSED THE ARCHITECTURE:")
print("   A fixed brain resident in RAM holds Wikipedia-scale knowledge with")
print(f"   {CEIL/14e9:.1f}x headroom. The 200 GB external memory bank in the earlier")
print("   design was over-engineered for anything short of web-scale long tail --")
print("   and it moved knowledge onto the SLOWEST component in the machine.")

# ---------------------------------------------------------------------------
hdr("4. WHERE THE KNOWLEDGE SHOULD LIVE — bandwidth comparison")

BW_RAM = 84 * GB          # decode-effective
BW_SSD = 3 * GB
print(f"{'tier':<28}{'bandwidth':>14}{'holds':>14}{'relative speed':>17}")
print(f"{'fixed brain (RAM)':<28}{BW_RAM/GB:>11.0f} GB/s{CEIL/1e9:>11.0f} Gb{BW_RAM/BW_SSD:>15.0f}x")
print(f"{'external bank (SSD)':<28}{BW_SSD/GB:>11.0f} GB/s{'~1600':>11} Gb{1:>15}x")
print("\nThe earlier design put 97% of parameters on the tier that is 28x slower.")
print("Dynamics-first puts the load-bearing knowledge on the 28x faster tier and")
print("uses SSD only for what genuinely exceeds the information ceiling.")

# ---------------------------------------------------------------------------
hdr("5. THE EXPERIMENT THAT DECIDES IT — knowledge bits per parameter")

print("Allen-Zhu baseline for a standard dense transformer : 2.0 bits/param")
print("Question nobody has published an answer to:")
print("  does a dynamic/Hebbian substrate (BDH/TTT-style) beat 2.0 bits/param?\n")
print(f"{'measured b/param':>18}{'verdict':>46}")
for b in (0.5, 1.0, 2.0, 3.0, 4.0):
    if b < 1.0:
        v = "KILL: dynamics is worse than plain weights"
    elif b < 2.0:
        v = "KILL for knowledge; keep dynamics for reasoning only"
    elif b == 2.0:
        v = "parity - premise not proven, not refuted"
    else:
        v = "PREMISE PROVEN, publishable"
    print(f"{b:>16.1f}  {v:>46}")

print("\nCapacity a fixed 6.5 GB brain would hold at each rate (ternary, 32.5B params):")
for b in (1.0, 2.0, 4.0):
    got = min(b * 32.5e9, CEIL)
    print(f"   {b:.1f} b/param -> {got/1e9:5.1f} Gbit"
          f" = {got/14e9:4.1f}x Wikipedia+textbooks")
