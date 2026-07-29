# 09 — Mistake Log

> Every mistake made on this project, who made it, how it was caught, and the guardrail that now
> prevents it. **Append-only. Never delete a row.**
>
> Read this before starting work. Most mistakes you are about to make are already here.

---

## How to use this

- **Before** writing code: scan the table for your area.
- **After** any error: add a row. Not adding a row is itself a violation.
- A mistake with no guardrail is not closed. "I'll be careful" is not a guardrail.

## Severity

| | Meaning |
|---|---|
| **S1** | Would have produced a wrong headline number, silently |
| **S2** | Would have wasted significant time or compute |
| **S3** | Local error, caught quickly |

---

## The log

| # | Sev | Who | Mistake | How caught | Guardrail now in place |
|---|---|---|---|---|---|
| M1 | S1 | Lead | Claimed `C(10000,100) ≈ 10^189`; true value `10^241.8` | Recomputing it in an executable script | All math in `verify_*.py`, output committed; no hand arithmetic in docs |
| M2 | S1 | Lead | Confused a **pattern count** with an **information capacity**. `10^241` patterns is 803 *bits*, not 10^241 bits | Re-derivation while writing the capacity doc | Capacity always stated in bits; `verify_capacity.py` prints both and labels them |
| M3 | S1 | Lead | Claimed "3.7× headroom" over Wikipedia by assuming 2.0 bits/param holds at ternary | Found the measured int4 collapse to 0.7 bits/param | Precision→capacity table is data, not assumption; ternary rows labelled EXTRAPOLATED |
| M4 | S1 | Lead | Missed that 2.0 bits/param requires **~1000 exposures per fact**; a 1-epoch run reaches 1.0 | Targeted search while chasing a different question | Exposure count is a required, recorded parameter of every capacity run; P1 gate states it |
| M5 | S1 | Lead | **Built the instrument around the rival hypothesis**: parameter-denominated metric, weights-based training, and a dataset with *no latent structure* — on which the thesis cannot win | User challenge | [`08_thesis_alignment.md`](08_thesis_alignment.md): substrate-byte denominator, Dataset B with latent structure, pre-registered predictions |
| M6 | S2 | Lead | Concluded the agent's resume tests were deficient because a break didn't fail them | Proving the restore is a no-op under a dropout-free model | Claims about *another's* test being wrong require demonstrating the break is observable first |
| M7 | S2 | Lead | Wrote two replacement tests that were **also** vacuous — they never exercised the path they claimed to test | Re-running the same break against my own tests | A new test must be shown to fail against the break that motivated it, before it is committed |
| M8 | S2 | Lead | Dispatched a Haiku agent for edits the lead then made directly; ~53k tokens on a no-op | Agent reported "nothing to commit" | [`07_model_tiering.md`](07_model_tiering.md) dispatch hygiene: check `git status` first; never edit a dispatched agent's paths |
| M9 | S1 | Haiku | Reported two implementations both returning **0.00** on a null model as "strong evidence the metric works" | Lead rejected it | **Zero is the trivial output.** Agreement on a null model is never evidence; the discriminating comparison is on a *trained* model |
| M10 | S3 | Lead | `.gitignore` pattern `data/` also matched `src/data/`, excluding source | `git add` refused the path | Ignore patterns are root-anchored (`/data/`) |
| M11 | S2 | Lead | Tokenizer design would have produced a **3.3M-token vocabulary** (~425M embedding params, larger than the 100M preset) by fusing 4-digit name suffixes | Computing vocab size before committing to it | Vocabulary size is computed and asserted before any model is configured from it |
| M12 | S3 | Lead | Own consistency test used a word absent from the closed corpus, so it failed | The tokenizer raised, correctly | Test inputs must be drawn from real generated text, never invented by hand |
| M13 | S2 | Sonnet | Quoted "89 passed" as a full-suite result when one of four test files had been excluded for slowness | Lead noticed the file count | A suite result must state which files ran; partial runs are labelled partial |
| M14 | S1 | Lead | Wrote a plateau detector thresholding an **arbitrary** gain ratio at 15%, with no stated relation to the quantity anyone cares about (capacity error). Also mislabelled a curve at 91% of ceiling a "textbook plateau" | Ran my own battery; it failed | Thresholds must be expressed in the units of the harm they prevent. Criterion is now "extrapolated remaining gain < 5% of measured value", i.e. a bound on capacity underestimate. Validated: predicted 9.98% remaining vs 10.0% ground truth |

---

## Patterns across these mistakes

Worth reading as a group, because the individual rows understate the shape of the problem.

**Six of thirteen are S1 — would have produced a wrong headline number with no error and no red
test.** This project's failure mode is not crashes. It is confident, plausible, wrong scalars.

**The lead made most of them, including all the worst ones.** M1–M5 are all lead errors, and M5 (the
instrument shaped around the rival hypothesis) is the most serious thing on the list. Seniority is not
protection; the process is.

**Four (M1–M4) share one root cause: a number asserted without being derived.** Every one was caught
by re-deriving it. Hence: *all quantitative claims live in executable scripts with committed output.*

**Two (M6, M7) are the lead being wrong about testing while criticising someone else's testing.**
Both were caught by applying the same standard to my own work that I applied to theirs. That symmetry
is the guardrail.

**M9 is the one to internalise if you read nothing else.** Two independent implementations agreeing
on `0.00` looks like strong corroboration and is worth nothing — a metric hardwired to return zero
produces exactly that agreement. **Agreement is only evidence on a value that is hard to produce by
accident.**
