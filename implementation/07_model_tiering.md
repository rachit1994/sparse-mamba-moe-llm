# 07 — Model Tiering: Who Does What

> Opus is the expensive resource and burns weekly limits fast. It is the **lead**, not the worker.
> This policy is binding on how work is assigned.

---

## The rule

**Opus decides. Sonnet builds. Haiku runs and reports.**

If a task does not require judgment that would change a decision, it does not belong on Opus.

---

## Tier assignment

| Tier | Model | Owns | Examples |
|---|---|---|---|
| **Lead** | **Opus** | judgment that changes decisions | architecture choices · gate verdicts · adjudicating whether a result is real · correcting errors · writing `reports/` verdicts · deciding what to kill |
| **Build** | **Sonnet** | implementation with tests | modules · pipelines · test suites · break-and-revert evidence · debugging · refactors |
| **Run** | **Haiku** | mechanical execution and reporting | running scripts and pasting raw output · sweeps · doc/index updates · formatting · scaffolding · collecting evidence for the lead to judge |

## What this changes about verification

"Green does not mean working" still holds — but **the lead does not need to press the buttons.**

Split it:

- **Haiku executes** the control, the mutation, the break-and-revert, and pastes **raw, unedited
  output**. Cheap, and mechanical execution is exactly what it is good at.
- **Opus adjudicates** that output: is the number real, does the control actually discriminate, is
  the conclusion warranted.

The rigour lives in *judging* the evidence, not in *generating* it. A Haiku agent that pastes a
failing test transcript is worth the same as an Opus one, at a fraction of the cost.

**Exception — the lead builds it directly when a silent-corruption trap is in play.** The tokenizer
was written by the lead because inconsistent name tokenisation would corrupt the headline number
with no error and no red test. That exception is narrow and must be justified in writing, not
assumed.

## Independent cross-checks stay, but get delegated

The two-implementations discipline (`tests/lead/`) is the strongest tool this project has. It does
**not** require Opus to write both. Assign the independent reimplementation to a **different model**
than the primary — a Haiku reimplementation of a Sonnet module is arguably *better* evidence than
two Opus ones, because the two are less likely to share a misunderstanding.

## Cost discipline for the lead

- Do not run long `Bash` sequences to gather evidence — delegate and read the summary.
- Do not write large docs that a spec-following model can write from a precise outline.
- Do not re-read files already read this session.
- Keep replies terse. Evidence and verdict, not narration.
- Batch independent work into parallel agents rather than serialising through the lead.

## Assigning a brief

Every brief in [`06_agent_briefs/`](06_agent_briefs/) names its tier. When spawning, pass the
matching `model` parameter. If a brief is unclear about tier, it defaults to **Sonnet** — never
Opus.

## Dispatch hygiene (learned the expensive way)

**Check current state before dispatching, and do not touch a dispatched agent's files.**

A Haiku agent was once sent to make a set of doc edits, and while it worked the lead made the same
edits directly in response to a stop-hook prompt. The agent completed ~20 minutes later against an
already-clean tree and reported "nothing to commit" — a correct report of a no-op that cost ~53k
tokens.

Rules that follow:

- Before spawning, run `git status` and confirm the work is not already done.
- Once an agent owns a path, the lead does not edit that path. If a hook or interrupt demands a
  commit, commit *what exists* and let the agent finish — do not race it.
- Prefer one agent per path, never overlapping ownership.
- For edits smaller than roughly a dozen lines, the lead doing it directly is cheaper than writing a
  brief, dispatching, and adjudicating. Delegation has fixed overhead; very small tasks are below it.
