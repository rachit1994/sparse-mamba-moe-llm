# 06 — Agent Briefs

Self-contained work packets. Each brief is written so an agent with **no prior context on this
project** can pick it up, do the work correctly, and report in a form the lead can verify.

## How to use a brief

1. Read [`../00_START_HERE.md`](../00_START_HERE.md) first. Always. It is the working contract.
2. Read your brief. It names your scope, your acceptance criteria, and the files you may not touch.
3. Read the spec your brief points at.
4. Do the work. Report with the template in `00_START_HERE.md` §5.

## Scope discipline

Briefs are written with **disjoint file ownership** so several agents can run in parallel without
conflicting. Your brief names exactly which paths you own.

If you need to change a file outside your scope: **stop and report.** Do not widen scope silently —
a concurrent agent is probably editing it.

## The rule that matters most

Every brief ends with the same requirement, and it is not boilerplate:

> **The deliverable is evidence the tests work, not that they pass.**

You must deliberately break the implementation, confirm the relevant test **fails**, then revert —
and report exactly what you broke and the real failure output. A test that does not fail when you
break its target is a broken test; fix it and say so.

A fully green suite with no demonstrated failures will be rejected without review. This is because
the project's output is a single scalar, and scalars are trivially faked
([`../02_testing_philosophy.md`](../02_testing_philosophy.md) §1).

## Current briefs

| Brief | Tier | Owns | Spec | Status |
|---|---|---|---|---|
| [B1 — dataset generator](B1_dataset_generator.md) | Build (Sonnet) | `src/data/generate.py`, `tests/test_data_generate.py` | [04](../04_golden_dataset.md) | done |
| [B2 — bits metric](B2_bits_metric.md) | Build (Sonnet) | `src/metrics/bits.py`, `tests/test_metrics_bits.py` | [04 §4](../04_golden_dataset.md) | done |
| [B3 — baseline model + training](B3_baseline_model.md) | Build (Sonnet) | `src/models/`, `src/train.py`, `tests/test_models_*.py` | [01 P1](../01_phases_and_gates.md) | done |

`src/data/schema.py` is **lead-owned and read-only for all agents.** It fixes the attribute
cardinalities that dataset entropy is derived from; a silent change there would not raise an
exception and would corrupt every downstream number.
