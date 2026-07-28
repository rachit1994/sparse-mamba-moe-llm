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
print("  ^ hard information ceiling: you cannot store more bits than you have.\n")

# MEASURED capacity per parameter, by precision (Allen-Zhu & Li).
# CORRECTION: an earlier version of this script assumed 2.0 bits/param at every
# precision, capped only by the storage bound. That was wrong. int4 does not hit
# a storage cap -- it suffers a MEASURED capacity collapse to 0.7 bits/param.
CAPACITY_BITS_PER_PARAM = {
    "fp16":    (2.0,  "measured"),
    "int8":    (2.0,  "measured - no loss vs fp16"),
    "int4":    (0.7,  "MEASURED COLLAPSE, not a storage cap"),
    "ternary": (0.7,  "EXTRAPOLATED from int4; unmeasured, trend is downward"),
}
print(f"{'precision':<10}{'B/param':>9}{'params':>10}{'cap b/p':>9}{'knowledge':>12}  note")
best = None          # best among MEASURED precisions only
best_speculative = None
for name, bpp in (("fp16", 2.0), ("int8", 1.0), ("int4", 0.5), ("ternary", 0.2)):
    cap, note = CAPACITY_BITS_PER_PARAM[name]
    p = RAM_WEIGHTS / bpp
    bits = min(cap * p, RAM_WEIGHTS * 8)
    measured = "EXTRAPOLATED" not in note
    if measured and (best is None or bits > best[1]):
        best = (name, bits)
    if best_speculative is None or bits > best_speculative[1]:
        best_speculative = (name, bits)
    print(f"{name:<10}{bpp:>9.2f}{p/1e9:>8.1f}B{cap:>9.1f}{bits/1e9:>10.1f} Gb  {note}")

print(f"\nBEST MEASURED CONFIGURATION : {best[0]} at {best[1]/1e9:.1f} Gbit")
print(f"Largest number in the table : {best_speculative[0]} at {best_speculative[1]/1e9:.1f} Gbit"
      f"  -- but that row is EXTRAPOLATED, so it is not a result.")
print("Planning uses the MEASURED figure. Ternary's apparent advantage rests")
print("entirely on an unmeasured assumption, and the measured trend is downward.")
print("KEY CONSEQUENCE: int4 holds 2x the PARAMETERS of int8 but LESS KNOWLEDGE")
print("  (9.1 Gbit vs 13.0 Gbit). Quantizing below int8 is counterproductive for")
print("  knowledge capacity. Ternary is worse still. Run capacity experiments at int8.")
print("\nEXPOSURE REQUIREMENT (the other thing that was missed):")
print("  2.0 bits/param requires each fact seen ~1000 times during training.")
print("  At ~100 exposures an undertrained model reaches only 1.0 bits/param.")
print("  => a 1-epoch training run CANNOT reach the baseline, and P1 would fail")
print("     its gate for reasons unrelated to architecture.")

# ---------------------------------------------------------------------------
hdr("3. IS THAT ENOUGH? — what the ceiling actually buys")

ACHIEVABLE = best[1]          # best MEASURED configuration, int8
STORAGE_CEIL = RAM_WEIGHTS * 8
targets = [
    ("English Wikipedia + textbooks", 14e9, "Allen-Zhu's own estimate for a 7B model"),
    ("30B-token web corpus (long tail)", 30e9 * 4 * 2, "~4 chars/token, ~2 bits/char compressed"),
    ("300B-token corpus (the Pile)", 300e9 * 4 * 2, ""),
]
print(f"Raw storage ceiling in {RAM_WEIGHTS/GB:.1f} GB : {STORAGE_CEIL/1e9:>6.1f} Gbit (unreachable)")
print(f"Best ACHIEVABLE (int8, measured)   : {ACHIEVABLE/1e9:>6.1f} Gbit\n")
for name, bits, note in targets:
    ratio = ACHIEVABLE / bits
    verdict = f"FITS ({ratio:.2f}x)" if ACHIEVABLE >= bits else f"SHORT by {bits/ACHIEVABLE:.1f}x"
    print(f"  {name:<36}{bits/1e9:>8.1f} Gb   {verdict}")
    if note:
        print(f"      {note}")

print("\n=> CORRECTED CONCLUSION (supersedes the '3.7x headroom' claim):")
print(f"   A fixed brain in {RAM_WEIGHTS/GB:.1f} GB reaches {ACHIEVABLE/1e9:.1f} Gbit at int8, versus")
print(f"   ~14 Gbit for Wikipedia+textbooks: {ACHIEVABLE/14e9:.2f}x. That is PARITY, NOT HEADROOM.")
print("   The earlier 3.7x figure assumed 2.0 bits/param at ternary; the measured")
print("   int4 collapse to 0.7 says that assumption was wrong and optimistic.")
print()
print("   The architecture decision still stands -- RAM is 28x faster than SSD and")
print("   is still the right tier for the load-bearing knowledge -- but the margin")
print("   is gone. The conditional overflow tier is now LIKELY to be needed, not")
print("   merely possible, and Phase 2 must measure where the boundary actually is.")

# ---------------------------------------------------------------------------
hdr("4. WHERE THE KNOWLEDGE SHOULD LIVE — bandwidth comparison")

BW_RAM = 84 * GB          # decode-effective
BW_SSD = 3 * GB
print(f"{'tier':<28}{'bandwidth':>14}{'holds':>14}{'relative speed':>17}")
print(f"{'fixed brain (RAM)':<28}{BW_RAM/GB:>11.0f} GB/s{ACHIEVABLE/1e9:>11.0f} Gb{BW_RAM/BW_SSD:>15.0f}x")
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

print("\nCapacity a fixed 6.5 GB brain would hold at each rate (int8, 6.5B params):")
for b in (1.0, 2.0, 4.0):
    got = min(b * 6.5e9, STORAGE_CEIL)
    print(f"   {b:.1f} b/param -> {got/1e9:5.1f} Gbit"
          f" = {got/14e9:4.2f}x Wikipedia+textbooks")
print("\n   Beating 2.0 b/param is not a vanity target: at int8 the fixed brain is")
print("   at 0.93x Wikipedia+textbooks. The premise has to WIN for the")
print("   dynamics-first architecture to stand without an overflow tier.")
