# 04 — Golden Dataset and Evaluation Questions

> The dataset **is** the experiment. If it is wrong, every downstream number is wrong and no amount
> of careful modelling recovers it.

---

## 1. Why synthetic, not real

The measurement is "how many bits of knowledge does this model store per parameter." That requires
knowing **exactly how many bits the dataset contains**. Two properties are non-negotiable:

| Requirement | Real corpora (Wikipedia, etc.) | Synthetic biographies |
|---|---|---|
| Exact entropy known | **No** — unknowable | **Yes** — by construction |
| Contamination-proof | **No** — in every pretraining set | **Yes** — generated from a private seed |
| Attribute cardinality controllable | No | Yes |
| Difficulty tunable | No | Yes |

Real-world benchmarks (MMLU etc.) cannot measure capacity because you cannot compute their entropy
and cannot prove the model has not seen them. **This follows Allen-Zhu & Li's methodology, which is
what produces the 2.0 bits/param baseline we are comparing against.** Deviating from it would make
the comparison invalid.

Real benchmarks still appear — as a **sanity tier** (§6), never as the capacity measurement.

---

## 2. Dataset construction

`N` synthetic individuals, each with `K` attributes drawn uniformly from known-cardinality sets.

| Attribute | Cardinality V | bits = log₂V |
|---|---:|---:|
| birth_date (day × year 1900–2099) | 73,000 | 16.16 |
| birth_city | 1,000 | 9.97 |
| university | 300 | 8.23 |
| major | 100 | 6.64 |
| employer | 1,000 | 9.97 |
| **Total per person** | | **50.97** |

**Dataset entropy is exact:**

```
H_total = N × Σ_j log₂(V_j) = N × 50.97 bits
```

Names are generated so that **no real person is representable**: `<Token><Token>-<4 digits>`, drawn
from a synthetic syllable inventory. This is the contamination proof and it is by construction.

**Sizing.** To find saturation you must bracket the model's capacity. At the 2.0 bits/param
baseline, a model with P parameters holds ~2P bits, so choose N such that H_total spans from well
below to well above 2P:

| Model params | Capacity @2 b/param | N for 0.25× | N for 1× | N for 4× |
|---:|---:|---:|---:|---:|
| 1 M | 2 Mbit | 9,810 | 39,238 | 156,954 |
| 10 M | 20 Mbit | 98,097 | 392,388 | 1,569,551 |
| 100 M | 200 Mbit | 980,969 | 3,923,877 | 15,695,507 |

**The saturation point is the measurement.** Train at several N and find where bits-recovered stops
rising. That plateau divided by parameter count is bits/param.

**Generate deterministically from `--seed`; never commit the data.** A 4M-person dataset is large;
the generator plus seed is the artifact.

---

## 3. Two formats, both required

**Training format — declarative.** Multiple paraphrases per person (Allen-Zhu found paraphrase
diversity materially affects extractable knowledge; with a single phrasing, knowledge is stored but
not extractable):

```
Kelmoran Vushiel-4417 was born on 12 March 1987.
Kelmoran Vushiel-4417 grew up in Tarnvale.
Kelmoran Vushiel-4417 studied Metallurgy at Brindlecourt University.
Kelmoran Vushiel-4417 works at Halcyon Dynamics.
```

**Evaluation format — interrogative, held out:**

```
Q: Where was Kelmoran Vushiel-4417 born?
A: Tarnvale
```

**Critical:** score **only the answer tokens**. Prompt tokens must be masked out
([02 §5](02_testing_philosophy.md)).

---

## 4. The metric

Bits recovered = dataset entropy minus the model's residual uncertainty:

```
bits_recovered(j) = Σ_people [ log₂(V_j) − CE_bits(model, correct_value) ]
bits_per_param    = Σ_j bits_recovered(j) / parameter_count
```

where `CE_bits` is cross-entropy over the answer tokens **in bits**.

```python
nats = F.cross_entropy(logits, targets, reduction="sum")   # PyTorch returns NATS
bits = nats / math.log(2)                                   # ← forgetting this = 1.44x error
```

Clamp per-attribute contributions at 0 — a model can be *worse* than chance on an item, but negative
"stored knowledge" is meaningless and would silently offset real gains elsewhere.

**Two assertions that must be live in code:**

1. `bits_recovered ≤ H_total` — you cannot recover more information than exists. If this fires, the
   metric is broken. This is the highest-value assertion in the project.
2. `bits_recovered(null_model) ≈ 0` — see [02 §2](02_testing_philosophy.md).

---

## 5. Splits

| Split | Purpose | Rule |
|---|---|---|
| **train** | 90% of people, all attributes | model sees these facts |
| **probe** | same people, **held-out paraphrasings** | measures extractable knowledge |
| **unseen-person** | 10% of people, never trained | **must score ~0** — this is a control, not a metric |
| **sealed** | separate seed, used only at phase gates | prevents tuning-to-test across phases |

**`unseen-person` scoring above ~0 means leakage.** Treat it as C1's twin: if the model "knows"
facts about people it never saw, the pipeline is broken.

**The sealed split may be looked at once per gate.** Not during development. This is the discipline
that keeps stage-to-stage comparisons honest.

---

## 6. LLM question sets

### Tier A — capacity probes (the measurement)
Generated from the synthetic dataset, one question per (person, attribute). Exact-match scored plus
cross-entropy. **This is the only tier that produces bits/param.**

### Tier B — extraction robustness
Does the model *know* the fact, or has it memorised one sentence? Same fact, five surface forms:

```
Where was Kelmoran Vushiel-4417 born?
Kelmoran Vushiel-4417's birthplace is ___
In which city did Kelmoran Vushiel-4417 grow up?
Tell me the hometown of Kelmoran Vushiel-4417.
Kelmoran Vushiel-4417 spent their childhood in ___
```

Large A-vs-B gaps mean knowledge is stored but not extractable — a real, publishable phenomenon and
a genuine risk to over-claiming capacity.

### Tier C — composition (tests the actual premise)
Facts never co-stated in training, requiring two retrievals plus a join:

```
Did Kelmoran Vushiel-4417 and Ashvern Toluma-9082 attend the same university?
Which of these people was born earliest?
List everyone who works at Halcyon Dynamics.
```

**Tier C is where "pattern interaction" (original Component 5) becomes measurable.** A lookup table
scores near chance here by construction. If a dynamic substrate beats a matched dense model on
Tier C, that is the strongest possible evidence for the premise — stronger than raw capacity.

### Tier D — sanity (real world, small)
A handful of real questions ("capital of France") purely to confirm the model is a functioning
language model. **Never used for capacity claims** — contamination is unprovable.

---

## 7. Failure modes specific to this dataset

| Trap | Consequence | Mitigation |
|---|---|---|
| Tokeniser splits names inconsistently | Answer inferable from token stats | Fixed vocabulary; verify names tokenise consistently |
| Attribute values correlate accidentally | Entropy overstated; model exploits correlation | Sample independently; assert empirical MI ≈ 0 |
| Same value dominates an attribute | Chance-level ≠ 1/V | Assert near-uniform empirical distribution |
| Paraphrase templates leak the answer | Inflated recovery | Template must contain no attribute-specific tokens |
| Train/probe overlap by accident | Contamination | Assert set intersection is empty |
| N too small for the model | Model memorises everything, no saturation | Bracket N per §2 |

Every row is an assertion that belongs in the generator's test suite, not a note.

---

## 8. Acceptance criteria for the generator

- [ ] `--seed` produces bit-identical output across runs
- [ ] Computed entropy matches `N × Σ log₂(V_j)` to <0.1%
- [ ] Empirical mutual information between attribute pairs ≈ 0
- [ ] Empirical attribute distributions near-uniform (χ² test passes)
- [ ] train ∩ probe = ∅ on paraphrasings; train ∩ unseen-person = ∅ on people
- [ ] No generated name matches any real-person pattern
- [ ] Names tokenise consistently under the chosen tokeniser
- [ ] Generates 4M people in reasonable time and streams to disk without OOM
- [ ] Round-trip serialise/deserialise is the identity
