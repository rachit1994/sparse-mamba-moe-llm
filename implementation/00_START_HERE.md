# 00 — START HERE (Agent Onboarding Contract)

**You are working on a research codebase where being subtly wrong is more expensive than being slow.**
Read this file completely before touching anything. It takes 5 minutes and prevents the three
failure modes that have already occurred in this project.

---

## 1. What this project is, in 60 seconds

We are testing one claim:

> Can a **fixed-size** neural substrate, whose knowledge lives in **dynamic activation patterns**
> rather than in ever-more parameters, store and use knowledge **more efficiently than ordinary
> weights**?

The measurable form of that claim, which is what you are actually building:

> **Knowledge bits stored per parameter.** The dense-transformer baseline is **2.0 bits/param**
> (Allen-Zhu & Li, ICLR 2025). If a dynamic substrate beats 2.0, the premise is proven and
> publishable. If it lands below 1.0, the premise is dead for knowledge.

Everything in `implementation/` serves that measurement. Background and the reasoning that got here:
[`../initial_research/10_dynamic_substrate.md`](../initial_research/10_dynamic_substrate.md).

**Target hardware is an M4 Mac mini with 16 GB unified memory.** You are probably working in a Linux
container. See §6 — this distinction matters and has caused errors already.

---

## 2. The three non-negotiables

These are not style preferences. Violating any one invalidates your work.

### N1 — A green test does not mean it works

The lead will **independently re-run and attack** everything you report. Assume that.

Before you report any result as working, you must have run **at least one test that would have
caught you if you were wrong.** Specifically:

- **Null-model control.** Run the same measurement on a randomly-initialised model. If your metric
  does not collapse, your metric is measuring the harness, not the model.
- **Mutation check.** Corrupt something that should matter (shuffle labels, zero a layer, permute
  the training data). If the number does not move, the number is not measuring what you claim.
- **Contamination check.** If a knowledge metric looks good, prove the model could not have seen
  the answers. For synthetic data this is by construction; state the construction.

A PR that says "all tests pass" without one of these gets rejected without review.

### N2 — Never report a number you did not produce

Paste the **command**, the **exit code**, and the **real output tail**. Never a summary of output you
did not see. Never a hypothetical transcript. If you could not run something, write
`UNVERIFIED:` in front of the claim. This is not optional and is trivially detected.

### N3 — Report the failure, don't fix the test

If a measurement comes back bad, that is a **result**, not a bug. This project has kill switches
precisely so that negative results are useful. Loosening an assertion, widening a tolerance, or
retrying until green **destroys the experiment**. If you believe a threshold is wrong, say so in
writing with a reason, and stop — do not change it.

Three of our seven gates produce a publishable result *when they fail*. A false green is worth
strictly less than an honest red.

---

## 3. How to pick up work

1. Read your brief in [`06_agent_briefs/`](06_agent_briefs/). It is self-contained: scope, exact
   acceptance criteria, files you may touch, files you may not.
2. Read [`02_testing_philosophy.md`](02_testing_philosophy.md) before writing any test.
3. Check [`03_edge_cases_and_scares.md`](03_edge_cases_and_scares.md) for known traps in your area.
   Most bugs you would introduce are already listed there.
4. Work only inside the file scope your brief names. If you need to change a file outside it, stop
   and report — do not widen scope silently.
5. Report using the template in §5.

## 4. Ground rules

| Rule | Why |
|---|---|
| **You may not edit a file you have not read** in this session | Prevents blind overwrites |
| **You may not modify anything in `initial_research/`** | It is the research record; corrections go through the lead |
| **You may not modify thresholds, gates, or acceptance criteria** | That is the experiment |
| **You may not add a dependency** without stating cost, what it replaces, and why stdlib is insufficient | Every dep must run on macOS/MLX later |
| **You may not commit large binaries or datasets** | Generate deterministically from a seed instead |
| **Determinism is required**: every script takes `--seed` and is reproducible | Non-reproducible results are not results |
| **No network at training time** | Contamination risk and it will not exist on the target box |

## 5. Reporting template

Every task ends with exactly this. If a section is genuinely empty write `none` — do not delete the
heading.

```
WHAT I DID
  <file:line> — <one clause>

HOW TO REPRODUCE
  $ <exact command>

VERIFICATION (real output only)
  $ <cmd>                    → exit 0, 42 passed
  null-model control:        <metric on random weights vs trained>
  mutation check:            <what I corrupted, how the metric moved>
  contamination check:       <why the model cannot have seen the answers>

WHAT I DID NOT TEST
  - <thing> — <why>

RESIDUAL RISK
  - <what could still be wrong>

NUMBERS
  <the actual measurement, with units>
```

## 6. Environment: container vs target hardware

**This has already caused one round of errors. Read carefully.**

| | Linux container (probably you) | M4 Mac mini (the target) |
|---|---|---|
| Purpose | build + validate harness correctness at **tiny scale** | run the **real** experiments |
| CPU/GPU | 4 CPU cores, no GPU | 10-core CPU, 10-core GPU, MLX |
| RAM | ~15 GB | 16 GB unified |
| Valid for | logic, shapes, determinism, unit tests, dataset generation | throughput, capacity, bits/param, wall-clock |
| **NOT valid for** | **any timing, throughput, or bits/param claim** | — |

**Rule: no performance number produced in the container may be reported as a result.** Mark it
`CONTAINER-ONLY: <number>`. The container exists to guarantee that when code reaches the Mac mini,
it is already known-correct — so that the only unknown left is hardware behaviour.

Network note: `pypi.org` is reachable; `download.pytorch.org` is policy-blocked. Install torch from
standard PyPI, not the pytorch index URL.

## 7. Where things live

```
initial_research/     research record — READ ONLY for agents
implementation/       specs, briefs, testing philosophy (this folder)
implementation/07_model_tiering.md   who does what: Opus leads, Sonnet builds, Haiku runs
src/                  the code you write
tests/                the tests you write
reports/              results pushed to main for the human — lead writes these
```

## 8. When to stop and escalate

Stop and report rather than proceeding on a guess if:

- A gate threshold appears wrong or unmeasurable
- Your result contradicts something in `initial_research/`
- You need to touch a file outside your brief's scope
- A measurement requires the Mac mini and you are in the container
- You cannot make a test fail when it should (N1) — **this is a red flag about the test, escalate it**

Escalating early is cheap. A confidently-wrong merged result costs weeks.
