# 12 — The Enforcement Harness

> **Every other document in this folder is advice. Advice failed.** 18 mistakes, 3 of them severity-1
> and originating in artifacts only the lead produces, 59% of build tokens spent on rework — all of it
> under a process that already had a testing philosophy, an edge-case catalogue, kill switches, and a
> tiering policy.
>
> This document specifies **checks that fail**, not reminders that are read. It is the highest-priority
> item in the project because nothing else survives without it.

---

## 1. Why the existing docs did not work

They were written for the reader's *intention*. Every one of them assumed the reader would remember,
at the right moment, under time pressure, while optimising for something else. That assumption is
false and the record proves it: I wrote `02_testing_philosophy.md` demanding mutation tests, then
wrote two vacuous tests myself (M7). I wrote `04_architecture.md` naming BDH and TTT, then dispatched
a brief that named neither (M17).

**The gap is not knowledge. It is that nothing ran.**

## 2. What the record says is mechanisable

Classifying all 18 logged mistakes by whether a check could have caught them *before any cost*:

| | Count |
|---|---:|
| Mechanically catchable | **17 / 18** |
| Partly (needs a judgement input, but the input is checkable) | 1 / 18 |
| Pure judgement, unmechanisable | **0 / 18** |

Not one mistake in this project required taste to prevent. Every one had a computable precondition
that was never computed.

## 3. The three artifact classes, and why only one was ever checked

| Artifact | Who writes it | Was it checked? | Defects originating here |
|---|---|---|---|
| **Code** | Sonnet | Yes — exhaustively (mutation, null controls, independent reimplementation) | M9, M13, M16 (all caught) |
| **Briefs** | **Lead** | **No** | **M17, M18** (both S1) |
| **Specs / thresholds** | **Lead** | **No** | **M3, M4, M5, M14** (three S1) |

**Code was gated to death. The lead's own artifacts were never gated at all.** Five of the six
severity-1 defects originate in an unchecked artifact. That is the whole finding.

---

## 4. `tools/preflight.py` — no dispatch without it

**Rule: an `Agent` call whose brief has not passed preflight is a process violation.** Exit non-zero
blocks the dispatch.

### Check P1 — brief completeness
The brief must be a committed file with every field present and non-empty:

```yaml
implements:        <doc §>            # e.g. 10_dynamic_substrate.md §3.1
paper:             <arXiv id + title> # e.g. arXiv:2509.26507 The Dragon Hatchling
equations:         <verbatim quote>   # the actual equations, not a paraphrase
owns:              <paths>
must_not_touch:    <paths>
tier:              haiku | sonnet
acceptance_breaks: <>= 3 break -> test-that-must-fail pairs>
token_estimate:    <from the measured table, §7>
```

*Catches M17 (no paper cited), M18 (no verbatim equations), M8 (owns/must_not_touch overlap).*

**Why `equations:` must be verbatim:** M17 happened because I paraphrased "BDH" into "Hebbian
associative memory". A paraphrase field would have passed. A quote field cannot — you either have the
paper's equations or you do not.

### Check P2 — degeneracy / triviality of the specified task
For any brief specifying a *mechanism with an objective*, require a stated argument, and where possible
an executable test, that the objective **cannot be satisfied by a fixed or closed-form solution**.

M18 in full: I specified square full-rank `theta_K`, `theta_V`, so `W* = theta_V @ theta_K^-1`
minimises `||Wk - v||^2` for *every* input. The state carried no information about the sequence. Two
different sequences converged to the same `W` (relative difference **6.37e-07**).

The check, generically:

```
Does a single input-independent parameter setting achieve the stated optimum?
  - rank argument:  if theta_K: R^d -> R^r with r < d, then W @ theta_K has rank <= r < d
                    and cannot equal a full-rank theta_V. Non-degenerate.
  - empirical:      run the inner objective on TWO unrelated sequences.
                    If the states converge to each other, the task is degenerate.
```

*This single check would have saved 311,301 tokens.*

### Check P3 — mistake-log collision
Grep `09_mistake_log.md` for the component name and every path in `owns:`. Print matching rows and
**require explicit acknowledgement in the brief** of how each is avoided.

*Catches repeats. M14 and M16 are the same error (a threshold with no relation to the harm) and the
second happened after the first was logged.*

### Check P4 — no duplicate or racing work
`git status` must show no uncommitted changes inside the brief's `owns:` paths, and no other live
agent may own an overlapping path.

*Catches M8 directly — 53,644 tokens on a no-op.*

### Check P5 — token pre-flight
`token_estimate` must be present and within 1.5× of the measured mean for its tier (§7). A build
estimated far above the mean must state why or be split.

---

## 5. `tools/faithfulness.py` — the gate that never existed

**Run before the correctness gate, on every delivered component.**

For each source file, resolve its brief's `implements:` and `paper:`, then require a
**property-to-test mapping**: each defining property claimed by the paper must name the test that
verifies it, and that test must exist and pass.

```
src/models/bdh.py   implements 10_dynamic_substrate.md §3.1 (arXiv:2509.26507)
  property: synaptic state, Hebbian, local          -> test_hebbian_locality          PASS
  property: sparse POSITIVE high-dim activations    -> test_sparsity_and_positivity    PASS
  property: attention EMERGES from correlation      -> test_emergent_association       MISSING  <-- FAIL
```

That last row is the real state of BDH right now, and it is exactly what an unaided read of "26 tests
pass" hides. **A component with an unmapped defining property is not done, regardless of its test
count.**

*Catches M17, M5, and the current BDH gap.*

## 6. `tools/claim_lint.py` — bans the specific fallacies already committed

Run over any report or commit message before it lands:

| Banned pattern | Mistake |
|---|---|
| Citing agreement between two implementations **at the trivial output value** (0.0, chance level) | M9 |
| Citing agreement without a **provenance check** that implementation B did not read A | M15 |
| A threshold whose units are not the units of the harm it prevents | M14, M16 |
| A suite result that does not enumerate the files that ran | M13 |
| A number from an extrapolated (`is_measured=False`) row used in a conclusion | M3 |
| A capacity claim without its exposure count and precision | M4 |
| A performance number produced in the container without `CONTAINER-ONLY` | H1 |

Each of these is a string/AST-level check, not a matter of taste.

## 7. Measured token model — replaces guessing

Actual costs from this session, to be used for `token_estimate`:

| Task type | Mean | Observed range |
|---|---:|---|
| Sonnet build | **~235,000** | 168,587 – 311,301 |
| Haiku run | **~44,000** | 28,340 – 53,644 |

Rules that follow from the numbers:

- **Rework, not verbosity, is the cost centre.** 59% of build tokens went to rework caused by bad
  specs. One preflight P2 check ≈ 311k tokens saved.
- **Sub-dozen-line edits: lead does them.** Delegation has ~44k fixed overhead.
- **Lead-authored source requires a written exception** naming the silent-corruption trap that
  justifies it. I wrote the tokenizer, plateau detector, and several verify scripts; only the
  tokenizer had a real justification.
- **Never dispatch two agents onto one path.** P4 enforces it.

## 8. Session-limit resilience (already working — keep it)

Five dispatches died mid-flight and cost nothing beyond their own tokens, because:

- every build commits and pushes **on completion**, never at the end of a phase
- every script takes `--seed` and is resumable
- a container restart during this session lost **zero** committed work

This is the one part of the process that held up under stress. Do not change it.

## 9. What remains genuine judgement

Being honest about the limits of mechanisation:

- **Is this the right experiment?** No check catches "the instrument is shaped around the rival
  hypothesis" (M5) in general. P2 and the faithfulness gate narrow it; they do not close it.
- **Is this mechanism worth building at all?** Taste.
- **Is a negative result real or an artifact?** The 107× figure for `dynamic.py` needed the M17
  insight to interpret, and no linter produces that.

Everything else on the list of 18 is enforceable, and therefore should be enforced rather than
remembered.

---

## 10. Build order

1. `tools/preflight.py` (P1, P3, P4 — pure string/git checks, no deps)
2. `tools/claim_lint.py` (independent, cheap)
3. `tools/faithfulness.py` (needs the brief schema from step 1)
4. P2 degeneracy checks (per-mechanism; the TTT rank argument is the first instance)

Steps 1–3 are mechanical and go to Sonnet. Step 4 needs the lead per mechanism, because the
degeneracy argument is mathematical.

**Until `preflight.py` exists, the lead runs its checks by hand and records having done so in the
dispatch.** A check performed manually is worth less than one that cannot be skipped, but it is worth
much more than a document nobody consults at the moment of decision.
