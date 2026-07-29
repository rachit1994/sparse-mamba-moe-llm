# 10 — Code Style: Write for the Reviewer

> **A human reviews every line of this code.** Optimise for their reading time, not your writing
> time. Code that is correct but unreviewable will be sent back.

---

## The one rule

**A reviewer must be able to check any claim without reading the whole file.**

That means: every input traceable to a source, every quantity computed by one named function, every
invariant asserted rather than described in prose.

## Required file structure for any script producing numbers

```
"""Docstring: THE QUESTION this file answers, then HOW TO REVIEW IT."""

# SECTION 1 -- CONSTANTS
#   Every input, each with a provenance comment naming the source.
#   Nothing derived here. A reviewer checks this section against the papers
#   and stops; if the constants are right, only the maths can be wrong.

# SECTION 2 -- PURE FUNCTIONS
#   One quantity per function. No printing. No module state.
#   Each independently checkable.

# SECTION 3 -- SELF-CHECKS
#   Assertions of identities the file depends on, run BEFORE any output.
#   A violated identity means nothing printed should be believed.

# SECTION 4 -- REPORT
#   Formatting only. No arithmetic. Calls Section 2.
```

Reference implementation: [`../initial_research/verify_substrate_bound.py`](../initial_research/verify_substrate_bound.py).

## Hard requirements

| Rule | Why |
|---|---|
| **No magic numbers.** Every literal is a named constant with a provenance comment | The reviewer must know where `0.138` came from without asking |
| **Separate data / computation / presentation** | Lets a reviewer check the constants without reading the formatting |
| **One quantity per function**, named after the quantity | `substrate_efficiency()` not `calc()` |
| **Type hints on every signature** | The reviewer should not infer types |
| **Docstrings say WHY, with `Raises:`** | The code already says what |
| **Assert invariants in code, not prose** | A claim in a comment is untested; an assertion is |
| **Mark extrapolated values `is_measured=False`** and exclude them from conclusions | Prevents an inferred number becoming a reported result (mistake M3) |
| **No bare `except`; no returning `None` on failure** | Silent failure is the project's main risk |
| **Functions under ~30 lines** | Longer means it is doing more than one thing |

## Naming

Names state what the thing **is for**, not what it is made of.

```python
# no
def calc(x, y): ...
data, info, result, temp, mgr, helper

# yes
def substrate_efficiency(scheme: EncodingScheme) -> float: ...
knowledge_bits_per_param, hopfield_retrievable_patterns
```

## Comments

Explain **why**, and especially why an obvious-looking alternative is wrong.

```python
# no
# loop over schemes
for scheme in SCHEMES:

# yes
# Extrapolated schemes are excluded deliberately: ternary shows the highest
# eta of any row, but that figure is inferred rather than observed, and
# planning against it would be planning against an assumption.
measured = [s for s in schemes if s.is_measured]
```

## Numbers in particular

This project's failure mode is a plausible wrong scalar. Therefore:

- State **units in the name**: `storage_bits_per_param`, not `storage`.
- State the **base**: `log2` in the name when it matters. Nats vs bits has already nearly cost us the
  headline number once (mistake M1/M9 territory).
- Any constant from a paper carries the **citation in the comment**.
- Any value that is inferred rather than measured is **flagged in the type**, not just the docstring.

## What gets sent back without review

- A magic number with no provenance
- Arithmetic inside a print statement
- A function that both computes and formats
- An invariant asserted in a comment rather than in code
- An extrapolated value used in a conclusion
- `except Exception:` without a stated reason

## For agents specifically

You are expected to behave as a senior engineer with ownership, not as a code generator:

- If the spec asks for something that will be unreviewable, **say so and propose the alternative**
  before building it.
- If you find a defect in existing code outside your scope, **report it** — do not silently work
  around it, and do not fix it in someone else's file.
- If your own test cannot fail when you break its target, **say so explicitly**. That is a finding,
  not an embarrassment (mistakes M6, M7 were the lead making exactly this error twice).
- Never widen a tolerance or loosen an assertion to get green. Report the red.
