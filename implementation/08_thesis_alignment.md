# 08 — Thesis Alignment: What We Are Actually Testing

> **This document supersedes the experiment design in [01](01_phases_and_gates.md) and
> [04](04_golden_dataset.md) where they conflict.** It exists because the lead let the build drift
> away from the thesis and toward a conventional LLM benchmark.

---

## 1. The drift, stated plainly

The thesis is:

> Intelligence emerges from **dynamic patterns** formed over a **fixed** substrate — not from
> knowledge packed into ever more weights. **Patterns are the carrier of knowledge; weights are
> infrastructure.**

What was actually built:

| Built | Thesis position |
|---|---|
| Decoder-only transformer, next-token prediction | conventional; the *control*, not the subject |
| Metric denominated in **parameters** | parameters are infrastructure, not the carrier |
| Dataset of **independent uniform** attributes | **no latent structure exists to form patterns over** |
| Training = SGD into weights | knowledge is supposed to live in dynamic state |

Three of four are the thing the thesis argues against. That is not "building the control first" — the
*instrument itself* was shaped around the weights hypothesis, and an instrument shaped around one
hypothesis cannot fairly adjudicate its rival.

## 2. The fatal flaw in the dataset

[`04_golden_dataset.md`](04_golden_dataset.md) specifies attributes sampled **independently and
near-uniformly**, and asserts empirical mutual information ≈ 0 as an acceptance test.

That was done for a good reason — exact, known entropy — and it has a consequence that was not
thought through:

**Independent uniform attributes contain no compressible structure.** Every fact is incompressible
relative to every other. There are no patterns to discover, no regularities to reuse, no compositions
to form. The dataset's entropy *is* its literal size.

On such data, the best any system can do is memorise. A pattern-forming substrate and a lookup table
converge to the same behaviour, because **there is nothing for pattern formation to exploit.**

We built the one dataset on which the thesis cannot win, and were about to run the decisive
experiment on it.

## 3. The corrected experiment: two datasets, opposite predictions

Keep the current dataset. Add a second. **The contrast between them is the experiment.**

### Dataset A — FLAT (the existing one)
Independent uniform attributes. Entropy `H_A = N × Σ log₂(V_j)`. No latent structure.
**Measures: raw memorisation capacity.**

### Dataset B — STRUCTURED (new; this is where the thesis lives)
Attributes generated from a small set of **latent factors**, so the data has real, discoverable
regularity while entropy stays exactly computable.

Construction sketch (the generator owns the details):

```
K latent archetypes, each a distribution over attribute values
each person: draw archetype k (log2(K) bits), then draw
             deviations from the archetype (a few bits each)

H_B = N × [ log2(K) + Σ_j deviation_bits_j ]     ← exact, and MUCH smaller than H_A
```

A system that **discovers the archetypes** needs ~`H_B` bits. A system that **memorises person by
person** needs ~`H_A` bits. The gap between them is the pattern-formation signal, and it is
measurable.

**The predictions, stated before running — this is what makes it a real experiment:**

| | Dense/weights arm | Dynamic/pattern arm |
|---|---|---|
| Dataset A (flat) | ~2.0 bits/substrate-unit | **≈ same** (nothing to exploit) |
| Dataset B (structured) | ~2.0 (memorises) | **> 2.0** (exploits structure) |

**If the dynamic arm does not beat the dense arm on B, the thesis is in serious trouble** — and that
is a real, publishable negative result. If it beats it on A but not B, something is wrong with the
measurement, because A is where there is least to gain.

Registering these predictions now is what stops a post-hoc story being fitted to whatever comes out.

## 4. The metric was denominated wrong

"Bits per **parameter**" silently assumes parameters are the store. For a system whose knowledge is
meant to live in *dynamic state*, that denominator begs the question.

**Corrected denominator: bits per byte of FIXED SUBSTRATE.**

```
substrate_bytes = weights + persistent state + index structures
                  (everything resident to answer a query)
bits_per_substrate_byte = bits_recovered / substrate_bytes
```

Why this is the fair comparison:

- It is **architecture-neutral.** A dense model spends its substrate on weights; a dynamic model
  spends some on weights and some on fast-weight/synaptic state. Both are charged for what they use.
- It matches the actual constraint: the Mac mini has **16 GB**, not "N parameters."
- It cannot be gamed by moving knowledge from weights into a large persistent state and calling the
  parameter count small — a real risk with the dynamic arm, and one the old metric would have missed.

Allen-Zhu's 2.0 bits/param remains the **calibration anchor** (P1 must still reproduce it, since a
harness that cannot recover a known result is untrustworthy), but the **comparison metric** between
arms is bits per substrate byte.

## 5. Borrowed LLM machinery: what stays and what must not leak in

The thesis is not "a better LLM." Conventional machinery is allowed only where it is genuinely
neutral infrastructure.

| Borrowed | Verdict | Reason |
|---|---|---|
| Tokenizer | **Keep** | Neutral I/O. Both arms use the identical one. |
| Next-token prediction objective | **Keep for the control arm only** | It is the dense baseline's native objective. The dynamic arm may need a different one — forcing it to use next-token would be assuming the conclusion. |
| Transformer backbone | **Control arm only** | It is what we are measuring *against*. |
| SGD into weights | **Control arm only** | The dynamic arm's knowledge must be able to enter via test-time state updates, not only gradients. |
| Bits-per-parameter | **Demoted** | Calibration anchor only. See §4. |
| Independent-uniform data | **Demoted** | Dataset A only. See §2, §3. |

**The rule:** any technique imported from today's LLMs must be justified as *neutral between the two
hypotheses*, or confined to the control arm. If a choice makes the dense arm's job easier and the
dynamic arm's harder, it is not infrastructure — it is a thumb on the scale.

## 6. What "from scratch" means here

Rethought rather than inherited:

- **Training.** The dynamic arm must be able to acquire knowledge **without a gradient step on base
  weights** — via test-time state updates (TTT/Titans-style) or Hebbian synaptic change (BDH-style).
  If it can only learn by SGD into weights, it is a transformer with extra steps and the thesis is
  untested.
- **Testing.** Tier C (composition) in [04 §6](04_golden_dataset.md) is promoted from nice-to-have to
  **primary**. A lookup table scores at chance on composition by construction; that is exactly the
  discriminating measurement, and Dataset B is what makes it meaningful.
- **Evaluation.** Report bits per substrate byte on **both** datasets. The A-vs-B *gap* is the
  headline, not either number alone.
- **Success criterion.** Not "beat 2.0". It is: **the dynamic arm's advantage grows with the amount
  of latent structure present.** That is a claim about a slope, and a slope is far harder to fake
  than a single scalar.

## 7. Status of work already done

Not wasted, but re-labelled:

| Artifact | Status |
|---|---|
| Dataset generator (flat) | **Valid** — this is Dataset A |
| Bits metric | **Valid** — needs a substrate-byte denominator added |
| TinyLM + training | **Valid as the CONTROL arm.** Not the subject |
| Tokenizer | **Valid** — neutral, shared |
| P0 controls | **Valid** — the apparatus works |
| P1 as specified in [01](01_phases_and_gates.md) | **Insufficient** — measures A only, in parameters only |

**Nothing built so far tests the thesis.** It is a validated instrument and a control arm, which is
genuine progress, but the subject of the experiment does not exist yet.
