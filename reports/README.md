# reports/ — Results for the Human

This folder is the human-facing interface to the project. Everything else in the repo is working
material.

**One report per completed phase, plus one on any kill-gate trigger.** Pushed to `main` when written.

## What a report is, and is not

A report states **what was measured, the actual number, how it could still be wrong, and the decision
taken.** It is a result with a verdict.

It is **not** a status update. "Working on P1" is not a report. If there is nothing measured, there
is no report.

## Format

```
# <NNN> — <Phase> — <one-line verdict>

Date · phase · commit

## The number
<the actual measurement, with units, and what it is compared against>

## Verdict
PROCEED / INVESTIGATE / STOP  — against which gate, and why

## How this was verified
<commands run, controls that fired, what the lead re-ran independently>

## How this could still be wrong
<residual risk, honestly>

## Decision
<what happens next>
```

## Index

| # | Phase | Verdict | Date |
|---|---|---|---|
| [001](001_P0_harness.md) | P0 harness correctness | INVESTIGATE — integration missing (superseded by 002) | 2026-07-28 |
| [002](002_P0_controls_pass.md) | P0 controls | **PASS** — apparatus reads the model, not the harness | 2026-07-28 |

## Standing rules

- **A green test suite is not a result.** No report is written on the strength of passing tests
  alone; every reported number carries a null-model control, a mutation check, and a contamination
  argument ([../implementation/02_testing_philosophy.md](../implementation/02_testing_philosophy.md)).
- **Numbers produced in the Linux container are labelled `CONTAINER-ONLY`** and are never reported as
  results. Only the M4 Mac mini produces valid performance and capacity numbers.
- **Negative results are reported the same way as positive ones.** Three of the seven research gates
  produce a publishable result when they fail; a project that can only report success will report
  success.
