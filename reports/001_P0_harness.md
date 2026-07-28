# 001 — P0 Harness Correctness — **INCOMPLETE: components verified, integration missing**

Date: 2026-07-28 · Phase: P0 · Commit: `d28cce1`

---

## The number

**None yet — and that is the correct outcome to report.**

P0 produces no capacity measurement. Its job is to establish that the measuring apparatus works
before anything is measured with it. Three of four acceptance criteria are met; the fourth is not.

| P0 criterion | Status |
|---|---|
| Controls demonstrated **failing when broken** | ✅ met, for all three components |
| Determinism verified by re-run | ✅ met |
| Overfit test drives loss ≈ 0 | ✅ met — 2.5137 → **0.018136** (chance = 2.4849) |
| Resume produces bit-identical weights | ✅ met |
| `bits_recovered ≤ entropy` assertion live | ✅ met |
| Analytic parameter count == actual, ≥3 configs | ✅ met — verified **four** independent ways |
| **Null model bits ≈ 0, unseen-person bits ≈ 0** | ❌ **NOT MET — cannot run** |

## Verdict

**INVESTIGATE — do not proceed to P1.**

## What blocks it

**The three components are individually verified but do not compose.** There is no
text-to-tokens path: `src/train.py` trains on randomly generated token IDs, not on the text the
dataset generator produces. So the end-to-end controls — null model and unseen-person leakage —
cannot be executed.

This is the classic integration failure: three green test suites, 139 passing tests, and a pipeline
that has never actually run end to end. Work to close it is in flight.

## How this was verified — independently, not on agent reports

Every number below was produced by re-running, not accepted from a report.

**Parameter count** (edge case S3 — the denominator of every capacity claim, and capable of
manufacturing a 2× difference from nothing). Made to agree across **four** independent sources:
the agent's analytic formula, the instantiated model, `count_parameters` in the metric, and a
formula I derived separately from standard GPT-2 block structure without reading their code.

```
tiny  analytic=     27,008  actual=     27,008  metric=     27,008  lead=     27,008
1m    analytic=  1,057,152  actual=  1,057,152  metric=  1,057,152  lead=  1,057,152
10m   analytic= 10,844,160  actual= 10,844,160  metric= 10,844,160  lead= 10,844,160
100m  analytic=101,558,272  actual=101,558,272  metric=101,558,272  lead=101,558,272
```

Agreement between only their formula and their model would have been vacuous.

**Metric** — cross-checked against `tests/lead/reference_bits.py`, a pure-NumPy implementation
written *before* the agent's code existed and confirmed untouched:

- six non-saturated mixed-probability cases agree to **<1e-6** (saturated cases would hide bugs)
- masking sound when answer logits are duplicated into the prompt region
- 5000-item accumulation drift **3.1e-07** relative
- I deleted the nats→bits conversion myself: **6 tests failed**, file restored byte-identical

**Dataset** — attacked beyond what the agent tested:

- 200,000 names generated, **200,000 unique**
- per-attribute empirical entropy recomputed from scratch, matches analytic maximum
- cross-seed independence: **0/500** positional agreement
- datasets are **nested under resize** (first 20 IDs at n=100 == first 20 at n=100,000)

## Two real defects found, one by an agent and one by me

**1. Name-collision bug (found by the dataset agent).** Main and sealed populations used two
independently-keyed permutations over the same domain — collision-free only *in expectation*. It
confirmed **2 real collisions** between two 100k populations against a birthday-bound expectation of
~3, then made uniqueness structural. Collisions would have made true entropy lower than
`N × BITS_PER_PERSON`, **silently inflating every bits/param number**.

**2. Capacity math error (found by me, in my own earlier work).** Two facts from Allen-Zhu & Li that
my first capacity pass missed, both making the picture worse:

- **Capacity collapses below int8.** int8 holds the full 2.0 bits/param; **int4 measures 0.7**. int4
  therefore holds *twice the parameters but less knowledge* (9.1 vs 13.0 Gbit). Capacity runs must
  use int8 or fp16 — never int4 or ternary.
- **2.0 bits/param requires ~1000 exposures per fact.** At ~100 it is 1.0. **A single-epoch run
  cannot reach the baseline** — P1 would have failed for reasons unrelated to architecture.

This supersedes my earlier "3.7× headroom" claim. Corrected: the fixed brain reaches **13.0 Gbit vs
~14 Gbit** for Wikipedia+textbooks — **0.93×, parity not headroom**. The dynamics-first architecture
still stands (RAM is 28× faster than SSD), but the margin is gone, the overflow tier is now *likely*
needed rather than merely possible, and beating 2.0 bits/param becomes **necessary** rather than a
vanity target.

## How this could still be wrong

- **The integration has never run.** Everything above tests components in isolation. Composition is
  where the remaining risk is concentrated.
- **No number here came from the target hardware.** All container-only; the M4 Mac mini has run
  nothing. Performance and capacity claims are unverified there.
- **One vacuous test, kept and labelled.** I deleted the RNG-state restore from the resume path and
  the tests still passed. I initially concluded the tests were deficient — **that was wrong**. The
  model is deliberately dropout-free, so training consumes no randomness and the restore is a proven
  no-op; the break is undetectable *by construction*, not by weak testing. The restore stays,
  guarded by a canary that fires the moment any RNG-consuming op is added.
- **Overfit threshold (0.025) is tuned** to one architecture, config, seed and thread count. Any
  change requires re-deriving it, not just re-running.

## Decision

1. Close the integration gap (tokenizer + end-to-end pipeline) — **in flight**.
2. Re-run P0 controls end to end. **Null model and unseen-person must both score ≈ 0.** If either
   does not, the metric is reading the harness and P1 must not start.
3. Only then P1, on the Mac mini, at **int8**, with **~1000 exposures per fact**.

**P0 is not complete and P1 has not started.** Reporting an incomplete phase as incomplete is the
point of having gates.
