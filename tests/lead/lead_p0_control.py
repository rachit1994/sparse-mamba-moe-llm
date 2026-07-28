"""Lead-owned independent P0 control: does the metric read the MODEL or the HARNESS?

Written deliberately WITHOUT reading src/integration.py, so that agreement between
this and the agent's pipeline is real evidence rather than a shared bug. Same
approach that caught problems in the metric: two implementations that cannot
share a misunderstanding.

The single most dangerous failure this project can have is a metric that produces
a beautiful number without consulting the model. A null (untrained) model must
score ~0 bits. If it does not, every downstream capacity number is measuring the
harness, and no amount of careful modelling later recovers it.

This file wires generate -> tokenize -> model -> bits by hand, using only the
public component APIs, and reports the three numbers that constitute the P0 gate.

Run:  python3 tests/lead/lead_p0_control.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generate import generate_people, render_probe_qa  # noqa: E402
from src.data.schema import ATTRIBUTE_CARDINALITY  # noqa: E402
from src.data.tokenizer import build_tokenizer  # noqa: E402
from src.models.config import ModelConfig  # noqa: E402
from src.models.tiny_lm import TinyLM  # noqa: E402

_LN2 = math.log(2.0)


def _score_answer_bits(model: TinyLM, ids: list[int], mask: list[bool]) -> float:
    """Cross-entropy in BITS over answer tokens only, computed from scratch.

    Deliberately does not import src.metrics.bits: this is the independent path.
    Next-token prediction, so position i's logits predict token i+1.
    """
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    target_mask = torch.tensor(mask[1:], dtype=torch.bool)
    if not bool(target_mask.any()):
        raise ValueError("no answer tokens after shifting for next-token prediction")

    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
    logits = logits[0].float()                       # (seq-1, vocab)

    logprobs_nats = torch.log_softmax(logits, dim=-1)
    picked = logprobs_nats[torch.arange(len(targets)), targets]
    total_nats = -float(picked[target_mask].sum())
    return total_nats / _LN2


def measure(
    *,
    n_people: int,
    seed: int,
    model: TinyLM,
    tokenizer,
    people_offset: int = 0,
) -> tuple[float, float]:
    """Return (bits_recovered, entropy_ceiling) over probe Q/A for n_people.

    bits_recovered = sum over (person, attribute) of max(0, log2(V) - CE_bits).
    Clamped at 0 per item: a model can be worse than chance, but negative stored
    knowledge is meaningless and would offset real gains elsewhere.
    """
    recovered = 0.0
    ceiling = 0.0
    people = itertools.islice(generate_people(n_people + people_offset, seed), people_offset, None)
    for person in people:
        for attribute, cardinality in ATTRIBUTE_CARDINALITY.items():
            question, answer = render_probe_qa(person, attribute, 0)
            ids, mask = tokenizer.encode_qa(question, answer)
            ce_bits = _score_answer_bits(model, ids, mask)
            per_item_ceiling = math.log2(cardinality)
            recovered += max(0.0, per_item_ceiling - ce_bits)
            ceiling += per_item_ceiling
    return recovered, ceiling


def main() -> int:
    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    config = ModelConfig(
        n_layer=2,
        n_head=2,
        d_model=64,
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        tie_embeddings=True,
    )
    n_people = 40

    print("=" * 70)
    print("LEAD INDEPENDENT P0 CONTROL  (CONTAINER-ONLY -- not a capacity result)")
    print("=" * 70)
    print(f"vocab_size={tokenizer.vocab_size}  n_people={n_people}  config d_model={config.d_model}")

    null_model = TinyLM(config)
    null_model.eval()
    null_bits, ceiling = measure(
        n_people=n_people, seed=101, model=null_model, tokenizer=tokenizer
    )

    print()
    print(f"  entropy ceiling over these probes : {ceiling:10.2f} bits")
    print(f"  NULL MODEL bits_recovered         : {null_bits:10.2f} bits")
    print(f"  as fraction of ceiling            : {100 * null_bits / ceiling:9.3f}%")
    print()

    # An untrained model is not exactly at chance: it has an arbitrary but fixed
    # bias over the vocabulary, and answer tokens are a small subset of it. The
    # honest tolerance is therefore "small fraction of ceiling", not "exactly 0".
    tolerance_frac = 0.05
    passed = null_bits <= tolerance_frac * ceiling
    print(f"  GATE: null bits <= {tolerance_frac:.0%} of ceiling"
          f" ({tolerance_frac * ceiling:.2f} bits) -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print()
        print("  *** The metric is reading the HARNESS, not the model. ***")
        print("  P1 must not start. Do not widen this tolerance to make it pass.")
    print()
    print("  NOTE: an untrained transformer over a 2818-token vocabulary is not")
    print("  uniform, so some apparent 'recovery' is expected. What must NOT")
    print("  happen is recovery comparable to a trained model's.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
