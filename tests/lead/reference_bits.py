"""Independent reference implementation of the bits-recovered metric.

LEAD-OWNED. This exists to cross-check src/metrics/bits.py, which is written by a
different agent using torch. This implementation deliberately:

  * uses only numpy + stdlib (no torch), so a torch-specific misunderstanding
    cannot be replicated identically in both,
  * is derived directly from the formula in implementation/04_golden_dataset.md
    section 4 without reading the other implementation,
  * is slow and obvious rather than fast and clever.

Two independent implementations agreeing on non-trivial inputs is real evidence.
A single implementation with a green test suite is not
(implementation/02_testing_philosophy.md section 1).

Formula:
    bits_recovered(j) = sum_people [ log2(V_j) - CE_bits(model, correct_value) ]
    bits_per_param    = sum_j bits_recovered(j) / parameter_count
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "log_softmax_rows",
    "answer_ce_bits",
    "bits_recovered",
    "chance_level_ce_bits",
]

_LN2 = math.log(2.0)


def log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable natural log-softmax over the last axis.

    Args:
        logits: array of shape (..., vocab).

    Returns:
        Log-probabilities in NATS, same shape.

    Raises:
        ValueError: if logits is not at least 1-D or contains non-finite values.
    """
    if logits.ndim < 1:
        raise ValueError(f"logits must be at least 1-D, got shape {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contains non-finite values")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def answer_ce_bits(
    logits: np.ndarray,
    targets: np.ndarray,
    answer_mask: np.ndarray,
) -> float:
    """Summed cross-entropy over answer tokens only, in BITS.

    The nats->bits conversion is the single most likely error in this project
    (implementation/03_edge_cases_and_scares.md S1): omitting it inflates results
    by exactly 1/ln2 = 1.4427. It happens exactly once, here.

    Args:
        logits: (seq, vocab) unnormalised scores.
        targets: (seq,) integer token ids.
        answer_mask: (seq,) boolean; True where the token is part of the answer.

    Returns:
        Total cross-entropy of the answer tokens, in bits. Non-negative.

    Raises:
        ValueError: on shape mismatch, out-of-range targets, or an all-False mask.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (seq, vocab), got shape {logits.shape}")
    seq, vocab = logits.shape
    if targets.shape != (seq,):
        raise ValueError(f"targets shape {targets.shape} does not match logits seq {seq}")
    if answer_mask.shape != (seq,):
        raise ValueError(f"answer_mask shape {answer_mask.shape} does not match logits seq {seq}")
    if not answer_mask.any():
        # A silent 0.0 here would read as "model knows nothing", which is a
        # different claim from "nothing was scored". Refuse to be ambiguous.
        raise ValueError("answer_mask selects no tokens; nothing to score")
    if targets.min() < 0 or targets.max() >= vocab:
        raise ValueError(f"targets out of range [0,{vocab}) : [{targets.min()},{targets.max()}]")

    logprobs_nats = log_softmax_rows(logits.astype(np.float64))
    picked_nats = logprobs_nats[np.arange(seq), targets]
    total_nats = -float(np.sum(picked_nats[answer_mask]))
    return total_nats / _LN2


def chance_level_ce_bits(cardinality: int, n_answer_tokens: int = 1) -> float:
    """Cross-entropy in bits of a model that is exactly at chance.

    Args:
        cardinality: number of equally likely values, V.
        n_answer_tokens: how many answer tokens carry the choice.

    Returns:
        log2(V) * n_answer_tokens.

    Raises:
        ValueError: if cardinality < 1 or n_answer_tokens < 1.
    """
    if cardinality < 1:
        raise ValueError(f"cardinality must be >= 1, got {cardinality}")
    if n_answer_tokens < 1:
        raise ValueError(f"n_answer_tokens must be >= 1, got {n_answer_tokens}")
    return math.log2(cardinality) * n_answer_tokens


def bits_recovered(
    cardinality: int,
    ce_bits_per_item: list[float],
    clamp_negative: bool = True,
) -> float:
    """Bits of knowledge recovered for one attribute across many people.

    Args:
        cardinality: V, the attribute's value-set size.
        ce_bits_per_item: model cross-entropy in bits for each person's answer.
        clamp_negative: clamp per-item contributions at 0. A model can be worse
            than chance, but negative "stored knowledge" is meaningless and would
            silently offset real gains elsewhere.

    Returns:
        Total bits recovered. Bounded above by len(items) * log2(V).

    Raises:
        ValueError: if cardinality < 1 or any cross-entropy is negative/non-finite.
    """
    if cardinality < 1:
        raise ValueError(f"cardinality must be >= 1, got {cardinality}")
    ceiling_per_item = math.log2(cardinality)
    total = 0.0
    for i, ce in enumerate(ce_bits_per_item):
        if not math.isfinite(ce) or ce < 0.0:
            raise ValueError(f"item {i}: cross-entropy must be finite and >= 0, got {ce}")
        gain = ceiling_per_item - ce
        if clamp_negative:
            gain = max(0.0, gain)
        total += gain

    # You cannot recover more information than exists. The highest-value
    # assertion in the project: if this fires, the metric is broken.
    ceiling_total = ceiling_per_item * len(ce_bits_per_item)
    if total > ceiling_total + 1e-6:
        raise AssertionError(
            f"bits_recovered {total} exceeds entropy ceiling {ceiling_total}"
        )
    return total
