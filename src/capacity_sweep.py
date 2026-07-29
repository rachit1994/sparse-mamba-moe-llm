"""P1 capacity-sweep harness: bits recovered vs. dataset size N.

THE QUESTION this file answers
    Train a FIXED-size dense transformer at increasing dataset sizes N (same
    architecture, same exposure count per fact, same seed) and find where
    bits_recovered(N) stops rising. That plateau, divided by parameter count,
    is what implementation/00_START_HERE.md calls "knowledge bits per
    parameter" -- the number that must land near Allen-Zhu & Li's 2.0
    baseline for the harness to be trusted (implementation/01_phases_and_gates.md
    P1). Divided by substrate bytes instead, it is the architecture-neutral
    metric implementation/08_thesis_alignment.md section 4 requires for
    comparison against a future dynamic-substrate arm.

    THE PLATEAU IS THE MEASUREMENT (03_edge_cases_and_scares.md S4). A curve
    that is still rising close to linearly has not saturated, and a ratio
    read off it measures dataset size, not model capacity. This file refuses
    to emit a capacity number in that case -- see detect_saturation and the
    "_at_plateau" functions below.

HOW TO REVIEW THIS FILE
    Section 1  CONSTANTS         every input, with provenance.
    Section 2  PURE FUNCTIONS    exposures -> steps, substrate bytes,
                                  saturation detection, the two gated capacity
                                  ratios. No I/O, no randomness, no model
                                  calls -- fully unit-testable on synthetic
                                  curves (tests/test_capacity_sweep.py).
    Section 3  DATA + EVAL       deterministic corpus construction and the
                                  two-tier evaluation (A: same phrasing family
                                  the model trained on; B: held-out probe
                                  phrasing). Wires generate -> tokenize ->
                                  model -> bits, following the verified path
                                  in tests/lead/lead_p0_control.py, through
                                  the production components
                                  (src.metrics.bits, src.models.tiny_lm)
                                  rather than that file's independent
                                  from-scratch cross-entropy.
    Section 4  ORCHESTRATION     one sweep point: build data, train via
                                  src.train.train_loop (unmodified), evaluate
                                  both tiers.
    Section 5  SWEEP + RESUME    drives Section 4 across --sizes; writes the
                                  results JSON after every point so a killed
                                  run resumes without redoing finished points.
    Section 6  SELF-CHECKS       invariants asserted before any sweep runs.
    Section 7  CLI / REPORT      argparse and JSON formatting only.

Every number this file's CLI prints is CONTAINER-ONLY until run on the target
Mac mini (implementation/00_START_HERE.md section 6) -- 4 CPU cores, no GPU,
no MLX. Timing and throughput here generalise to nothing; correctness does.

Run: python3 -m src.capacity_sweep --seed 0 --exposures 30 --sizes 16,32,64,128 --out /tmp/p1_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from src.data.generate import Person, generate_people, render_probe_qa, render_training_text
from src.data.schema import ATTRIBUTES, dataset_entropy_bits
from src.data.tokenizer import PAD, Tokenizer, build_tokenizer
from src.metrics.bits import (
    answer_cross_entropy_bits,
    bits_per_parameter,
    bits_recovered_per_attribute,
    count_parameters,
)
from src.models.config import ModelConfig, analytic_param_count
from src.models.tiny_lm import TinyLM, causal_lm_shift
from src.train import train_loop

__all__ = [
    "UnsaturatedCurveError",
    "SaturationVerdict",
    "TierResult",
    "SweepPointResult",
    "SweepConfig",
    "training_steps_for_exposures",
    "substrate_bytes",
    "detect_saturation",
    "bits_per_parameter_at_plateau",
    "bits_per_substrate_byte_at_plateau",
    "run_sweep_point",
    "build_report",
    "run_self_checks",
    "main",
]

# =============================================================================
# SECTION 1 -- CONSTANTS
# Every input to this file, with provenance. Nothing is derived here.
# =============================================================================

#: Blocking condition 2 (precision, mistake M3): capacity collapses below
#: int8, and this project's precision/capacity table
#: (initial_research/verify_substrate_bound.py SCHEMES) was only ever measured
#: down to int8. This file trains and evaluates in fp32 only -- no dtype
#: parameter is exposed anywhere below, which is the only way to guarantee a
#: ternary or int4 capacity number can never be emitted from this file.
#: Source: implementation/01_phases_and_gates.md P1 precision box.
TRAINING_DTYPE: Final[torch.dtype] = torch.float32

#: Storage bytes per parameter under TRAINING_DTYPE, grounded via torch.finfo
#: (IEEE 754 single precision = 32 bits) rather than a hardcoded literal.
BYTES_PER_PARAM: Final[int] = torch.finfo(TRAINING_DTYPE).bits // 8

#: Blocking condition 1 (exposures, mistake M4): the 2.0 bits/param baseline
#: requires each fact to be seen ~1000 times; ~100 exposures caps recovery
#: near 1.0 bits/param. Source: implementation/01_phases_and_gates.md P1
#: exposure box (Allen-Zhu & Li).
ALLEN_ZHU_EXPOSURES_FOR_2_BITS: Final[int] = 1000
ALLEN_ZHU_BITS_AT_LOW_EXPOSURE: Final[float] = 1.0

#: "Assert exposures >= 1" -- implementation/01_phases_and_gates.md P1 brief,
#: blocking condition 1. Zero exposures means zero training steps: a
#: meaningless "capacity" measurement of an untrained model.
MIN_EXPOSURES: Final[int] = 1

#: Largest tolerable underestimate of capacity, expressed as a fraction of the
#: bits_recovered value read off the curve's last point -- not an arbitrary
#: ratio threshold. A curve counts as saturated when the geometric-decay
#: extrapolation of its still-unmeasured remaining gain (see detect_saturation)
#: is under this fraction of what was already measured. At 5%, a capacity
#: reported as 2.00 bits/param could truly be up to ~2.10 -- comfortably
#: inside P1's own +/-25% gate, so saturation error is not the dominant term
#: (matches tests/lead/lead_plateau_check.py's independently-derived value and
#: rationale, MAX_REMAINING_FRACTION; that file's own history is worth
#: reading: its first version thresholded a bare ratio of marginal gains at
#: 15%, which mislabelled a curve at 90.9% of its true asymptote as
#: "saturated" -- an arbitrary ratio has no stated relationship to the
#: quantity that actually matters, the capacity error. This file learned that
#: lesson from the same discovery rather than re-deriving it independently).
MAX_REMAINING_FRACTION: Final[float] = 0.05

#: Fewest sweep points needed to judge saturation: estimating the geometric
#: decay of marginal gains by least-squares regression (_log_linear_decay_ratio)
#: needs at least 3 gains -- 2 gives a perfect but meaningless fit (any two
#: points define a line) -- hence 4 points. Matches
#: tests/lead/lead_plateau_check.py's independently-chosen MIN_POINTS_FOR_JUDGEMENT.
MIN_SWEEP_POINTS_FOR_SATURATION: Final[int] = 4

#: Every attribute's TRAIN and PROBE template pool is asserted (at import
#: time, in src/data/generate.py, immediately after _TRAIN_TEMPLATES /
#: _PROBE_TEMPLATES are defined) to have at least 5 entries. That assertion
#: is a publicly enforced invariant of a USE-ONLY file, not merely an
#: observed count -- relying on "at least 5 valid template_idx values" is
#: therefore safe without reaching into the generator's private pool-size
#: internals or hardcoding an exact count that could silently go stale.
MIN_GUARANTEED_TEMPLATES_PER_ATTRIBUTE: Final[int] = 5

#: Row length for the training corpus (Section 3). Must accommodate the
#: longest generated biography without truncation. Measured empirically:
#: across 5,000 sampled biographies (seed=42, the generator's default
#: template RNG), token length ranged 59-77 (5 declarative sentences,
#: src.data.tokenizer's word-level closed vocabulary). 96 gives ~25% headroom
#: over the observed maximum. This is a starting point, not a silent
#: assumption: _build_training_corpus (Section 3) asserts every row fits and
#: raises, naming the offending person, rather than truncating, if it does
#: not -- so a future template change that lengthens biographies fails loudly
#: instead of corrupting training data.
DEFAULT_BLOCK_SIZE: Final[int] = 96


# =============================================================================
# SECTION 2 -- PURE FUNCTIONS
# One quantity per function. No I/O, no randomness, no model calls. Each is
# independently unit-testable (tests/test_capacity_sweep.py).
# =============================================================================


def training_steps_for_exposures(exposures: int, n_train_examples: int, batch_size: int) -> int:
    """Optimizer steps giving each training example ~`exposures` epoch-passes.

    This is the one place "exposures" becomes a step count (blocking
    condition 1 / mistake M4), so every sweep result can be traced back to
    how many times its facts were shown during training.

    src.train.ToyBatcher (USE, DO NOT MODIFY) reshuffles into a fresh epoch
    once it runs out of examples and drops the undersized remainder of the
    outgoing epoch, so `steps * batch_size / n_train_examples` is the number
    of epoch-equivalents to within one dropped partial batch per epoch --
    negligible once `exposures * n_train_examples` is large relative to
    `batch_size`, which every sweep point in this file satisfies.

    Args:
        exposures: target number of times each training example is seen.
            Must be >= MIN_EXPOSURES.
        n_train_examples: examples in one epoch (this sweep point's N). Must
            be a positive int.
        batch_size: examples per optimizer step. Must be a positive int.

    Returns:
        Total optimizer steps, >= 1.

    Raises:
        ValueError: if exposures < MIN_EXPOSURES, or if n_train_examples or
            batch_size is not a positive int.
    """
    if exposures < MIN_EXPOSURES:
        raise ValueError(f"exposures must be >= {MIN_EXPOSURES}, got {exposures}")
    if n_train_examples <= 0:
        raise ValueError(f"n_train_examples must be positive, got {n_train_examples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return math.ceil(exposures * n_train_examples / batch_size)


def substrate_bytes(
    parameter_count: int,
    *,
    bytes_per_param: int = BYTES_PER_PARAM,
    persistent_state_bytes: int = 0,
    index_bytes: int = 0,
) -> int:
    """Total resident bytes needed to answer a query.

    substrate_bytes = weights + persistent state + index structures
    (implementation/08_thesis_alignment.md section 4): the architecture-
    neutral denominator. A dense model spends its whole budget on weights; a
    dynamic model may spend some on fast-weight/synaptic state; both are
    charged for what they actually use, and neither can hide capacity by
    moving it out of the parameter count.

    For THIS file's control arm -- a plain decoder-only transformer, no
    test-time state, no external index -- persistent_state_bytes and
    index_bytes are 0. That is stated explicitly as an accepted parameter
    rather than hardcoded away, so a future dynamic-arm caller can supply
    non-zero values through this same function without it changing shape.

    Args:
        parameter_count: as returned by src.metrics.bits.count_parameters.
            Must be positive.
        bytes_per_param: storage bytes per parameter under the run's dtype.
            Defaults to BYTES_PER_PARAM (TRAINING_DTYPE, this file's only
            allowed precision). Must be positive.
        persistent_state_bytes: bytes of dynamic state resident between
            queries. N/A for this control arm (0 by default). Must be
            non-negative.
        index_bytes: bytes of an external index structure. N/A for this
            control arm (0 by default). Must be non-negative.

    Returns:
        Total substrate bytes.

    Raises:
        ValueError: if parameter_count or bytes_per_param is not positive, or
            if persistent_state_bytes/index_bytes is negative.
    """
    if parameter_count <= 0:
        raise ValueError(f"parameter_count must be positive, got {parameter_count}")
    if bytes_per_param <= 0:
        raise ValueError(f"bytes_per_param must be positive, got {bytes_per_param}")
    if persistent_state_bytes < 0:
        raise ValueError(f"persistent_state_bytes must be non-negative, got {persistent_state_bytes}")
    if index_bytes < 0:
        raise ValueError(f"index_bytes must be non-negative, got {index_bytes}")
    return parameter_count * bytes_per_param + persistent_state_bytes + index_bytes


def _bits_per_substrate_byte(total_bits_recovered: float, substrate_bytes_value: int) -> float:
    """Bits recovered per byte of resident substrate.

    Mirrors src.metrics.bits.bits_per_parameter's contract exactly, with
    substrate bytes as the denominator instead of parameter count. No
    equivalent exists in bits.py (USE, DO NOT MODIFY), so it is defined here.
    Private: the only public entry points are the "_at_plateau" functions
    below, which gate this on saturation first -- a caller cannot obtain a
    bits/substrate-byte number without going through the saturation check.

    Raises:
        ValueError: if substrate_bytes_value is not positive, or if
            total_bits_recovered is negative.
    """
    if substrate_bytes_value <= 0:
        raise ValueError(f"substrate_bytes_value must be positive, got {substrate_bytes_value}")
    if total_bits_recovered < 0:
        raise ValueError(f"total_bits_recovered must be non-negative, got {total_bits_recovered}")
    return float(total_bits_recovered) / float(substrate_bytes_value)


@dataclass(frozen=True, slots=True)
class SaturationVerdict:
    """Result of a saturation judgement.

    Attributes:
        is_saturated: whether capacity may be read off the curve's last
            point.
        decay_ratio: estimated per-interval geometric decay factor of the
            marginal gains (< 1.0 means gains are shrinking; >= 1.0 means
            they are not, and no finite asymptote exists to extrapolate).
        remaining_fraction: the extrapolated, still-unmeasured gain beyond
            the last point, as a fraction of the value already measured --
            this IS the capacity underestimate a caller would be exposed to
            by reading the curve now. math.inf when gains do not decay.
        reason: human-readable justification, for the report.
    """

    is_saturated: bool
    decay_ratio: float
    remaining_fraction: float
    reason: str


def _log_linear_decay_ratio(gains: list[float]) -> float:
    """Per-interval geometric decay factor of marginal gains, via least-squares
    regression of log(gain) against interval index.

    A saturating curve's marginal gains shrink geometrically: gain_i ~=
    gain_0 * r^i for some ratio r < 1, so log(gain_i) is linear in i with
    slope log(r). Fitting that line by least squares uses every interval's
    gain, not only the first and the last
    (tests/lead/lead_plateau_check.py's independently chosen estimator, which
    reads the decay off the two endpoints only) -- a genuinely different
    estimator, less sensitive to noise landing on any single interval, at the
    cost of assuming the decay is close to geometric across the whole curve
    rather than only at its ends. Agreement between the two on the same
    battery (tests/test_capacity_sweep.py) is checked precisely because the
    estimators differ (implementation/02_testing_philosophy.md section 1;
    mistake M9).

    Returns:
        The fitted decay ratio, or 1.0 ("no decay") if there are fewer than
        two gains to fit a slope through, if any gain is non-positive (a
        geometric model on a non-positive gain is undefined), or if every
        interval index coincides (degenerate regression).
    """
    if len(gains) < 2 or any(gain <= 0 for gain in gains):
        return 1.0
    indices = list(range(len(gains)))
    log_gains = [math.log(gain) for gain in gains]
    mean_index = sum(indices) / len(indices)
    mean_log_gain = sum(log_gains) / len(log_gains)
    covariance = sum((i - mean_index) * (lg - mean_log_gain) for i, lg in zip(indices, log_gains))
    variance = sum((i - mean_index) ** 2 for i in indices)
    if variance == 0:
        return 1.0
    return math.exp(covariance / variance)


def detect_saturation(
    n_values: Sequence[float],
    bits_values: Sequence[float],
    *,
    threshold: float = MAX_REMAINING_FRACTION,
) -> SaturationVerdict:
    """Decide whether capacity may be read off a bits-recovered-vs-N curve.

    THE PLATEAU IS THE MEASUREMENT (03_edge_cases_and_scares.md S4): if bits
    keep rising close to linearly as N grows, saturation was never reached,
    and bits/parameter measures dataset size, not model capacity. This
    function is the guard that stops that number from being reported.

    Criterion: extrapolate the still-unmeasured remaining gain as the tail of
    a geometric series in the observed decay of marginal gains (bits per
    additional person, per swept interval), and call the curve saturated
    when that remainder is a small fraction (< `threshold`) of the value
    already measured. This directly bounds the quantity that matters --
    how wrong a capacity number read off the last point could be -- rather
    than thresholding an arbitrary ratio of marginal gains with no stated
    relationship to that error (see MAX_REMAINING_FRACTION's provenance
    comment for the concrete counter-example that rules the ratio approach
    out: a curve at 90.9% of its true asymptote, which a bare-ratio criterion
    mislabels as saturated).

    Args:
        n_values: dataset sizes N, strictly increasing.
        bits_values: bits recovered at each N, same length as `n_values`.
            Must be non-decreasing (see Raises) and end on a positive value.
        threshold: remaining_fraction below which the curve counts as
            saturated.

    Returns:
        A SaturationVerdict.

    Raises:
        ValueError: if `n_values`/`bits_values` have mismatched lengths,
            fewer than MIN_SWEEP_POINTS_FOR_SATURATION points, non-increasing
            `n_values`, any interval where bits_values decreases (a capacity
            curve should never fall as N grows for a fixed model and fixed
            exposure count; a decrease signals training instability, not
            saturation, and must be investigated rather than silently
            interpreted), a non-positive final value, or a first-interval
            gain that is not positive (the curve never started rising, so
            there is nothing to extrapolate).
    """
    if len(n_values) != len(bits_values):
        raise ValueError(
            f"n_values and bits_values must be the same length, got "
            f"{len(n_values)} and {len(bits_values)}"
        )
    if len(n_values) < MIN_SWEEP_POINTS_FOR_SATURATION:
        raise ValueError(
            f"need >= {MIN_SWEEP_POINTS_FOR_SATURATION} points to judge saturation "
            f"(estimating decay needs at least 3 gains), got {len(n_values)}"
        )
    if any(b <= a for a, b in zip(n_values, n_values[1:])):
        raise ValueError(f"n_values must be strictly increasing, got {list(n_values)}")
    if bits_values[-1] <= 0:
        raise ValueError(f"final bits_values must be positive, got {bits_values[-1]}")

    gains = [
        (bits_values[i] - bits_values[i - 1]) / (n_values[i] - n_values[i - 1])
        for i in range(1, len(n_values))
    ]
    if any(gain < 0 for gain in gains):
        raise ValueError(
            f"bits_values must be non-decreasing in N for a fixed model and exposure "
            f"count; got interval gains {gains} for n_values={list(n_values)} -- a "
            f"decrease signals training instability, not saturation, and must be "
            f"investigated rather than silently interpreted"
        )
    if gains[0] <= 0:
        raise ValueError(
            f"first-interval gain must be positive, got {gains[0]}; a curve that never "
            f"started rising has no growth regime to extrapolate from"
        )

    decay = _log_linear_decay_ratio(gains)
    if decay >= 1.0:
        return SaturationVerdict(
            is_saturated=False,
            decay_ratio=decay,
            remaining_fraction=math.inf,
            reason="marginal gains are not decaying; no finite asymptote to extrapolate",
        )

    # Remaining gain: the geometric tail beyond the last measured point, at
    # the fitted per-interval decay, scaled by the width of the final
    # interval so the extrapolation is in units of bits, not "per interval".
    last_step = n_values[-1] - n_values[-2]
    remaining_absolute = gains[-1] * decay / (1.0 - decay) * last_step
    remaining_fraction = remaining_absolute / bits_values[-1]

    saturated = remaining_fraction < threshold
    reason = (
        f"gains decay {decay:.3f}x per interval; extrapolated remaining gain is "
        f"{remaining_fraction:.1%} of the measured value "
        f"({'within' if saturated else 'over'} the {threshold:.0%} tolerance)"
    )
    return SaturationVerdict(
        is_saturated=saturated, decay_ratio=decay, remaining_fraction=remaining_fraction, reason=reason
    )


class UnsaturatedCurveError(ValueError):
    """Raised when a capacity ratio is requested from a curve that has not plateaued.

    03_edge_cases_and_scares.md S4: bits/parameter (or bits/substrate-byte)
    computed from an unsaturated curve measures dataset size, not model
    capacity -- the single most likely way this project reports a wrong
    headline number. The check is enforced structurally inside every
    capacity-ratio function in this file: there is no code path that returns
    a number without first calling detect_saturation.
    """


def bits_per_parameter_at_plateau(
    n_values: Sequence[float],
    bits_values: Sequence[float],
    parameter_count: int,
    *,
    threshold: float = MAX_REMAINING_FRACTION,
) -> float:
    """Plateau bits_recovered / parameter_count, gated on saturation.

    The plateau value is bits_values at the largest swept N -- valid only
    once detect_saturation confirms the curve has actually flattened there.
    Calibration anchor against Allen-Zhu & Li's 2.0 bits/param
    (implementation/00_START_HERE.md section 1); the comparison metric
    between architectures is bits_per_substrate_byte_at_plateau below
    (implementation/08_thesis_alignment.md section 4).

    Raises:
        UnsaturatedCurveError: if detect_saturation says the curve has not
            plateaued.
        ValueError: propagated from detect_saturation (malformed curve) or
            from src.metrics.bits.bits_per_parameter (parameter_count <= 0).
    """
    verdict = detect_saturation(n_values, bits_values, threshold=threshold)
    if not verdict.is_saturated:
        raise UnsaturatedCurveError(
            f"curve is not saturated ({verdict.reason}); refusing to report bits/parameter "
            f"from an unsaturated curve, which would measure dataset size N, not model "
            f"capacity (03_edge_cases_and_scares.md S4). Sweep to larger N."
        )
    return bits_per_parameter(bits_values[-1], parameter_count)


def bits_per_substrate_byte_at_plateau(
    n_values: Sequence[float],
    bits_values: Sequence[float],
    substrate_bytes_value: int,
    *,
    threshold: float = MAX_REMAINING_FRACTION,
) -> float:
    """Plateau bits_recovered / substrate_bytes, gated on saturation.

    Identical gating to bits_per_parameter_at_plateau; substrate bytes
    (weights + persistent state + index, see substrate_bytes()) instead of
    raw parameter count as the denominator -- the architecture-neutral
    comparison metric (implementation/08_thesis_alignment.md section 4).

    Raises:
        UnsaturatedCurveError: if the curve has not plateaued.
        ValueError: propagated from detect_saturation, or if
            substrate_bytes_value <= 0 or bits_values[-1] < 0.
    """
    verdict = detect_saturation(n_values, bits_values, threshold=threshold)
    if not verdict.is_saturated:
        raise UnsaturatedCurveError(
            f"curve is not saturated ({verdict.reason}); refusing to report "
            f"bits/substrate-byte from an unsaturated curve (03_edge_cases_and_scares.md "
            f"S4). Sweep to larger N."
        )
    return _bits_per_substrate_byte(bits_values[-1], substrate_bytes_value)


# =============================================================================
# SECTION 3 -- DATA + EVAL (impure: randomness, tokenisation, model calls)
# Wires generate -> tokenize -> model -> bits, following the verified path in
# tests/lead/lead_p0_control.py, through the production src.metrics.bits /
# src.models.tiny_lm components.
# =============================================================================


def _train_family_prompt_answer(sentence: str, value: str) -> tuple[str, str]:
    """Split a TRAIN-family declarative sentence into (prompt, answer) at the value.

    Tier A evaluation (blocking condition 4): score the model on a sentence
    drawn from the SAME template pool it trained on
    (src.data.generate._TRAIN_TEMPLATES), as opposed to Tier B's structurally
    disjoint PROBE templates. Every TRAIN template places {value} as the last
    word(s) before the final period (implementation/04_golden_dataset.md
    section 3's examples; enforced by src/data/generate.py's template
    construction), so locating the value substring and splitting there
    recovers exactly the (prompt, answer) shape
    src.data.tokenizer.Tokenizer.encode_qa expects -- using only the public
    render_training_text output, never the private template dict.

    Args:
        sentence: one declarative sentence for a person/attribute, as
            returned by src.data.generate.render_training_text.
        value: the attribute's true value (person.attributes[attribute]).

    Returns:
        (prompt, answer) where answer == value verbatim and prompt is
        everything in `sentence` before it.

    Raises:
        ValueError: if `value` does not occur in `sentence` exactly once.
            Invented attribute values never collide with template or name
            vocabulary by construction (src/data/generate.py module
            docstring); a count != 1 means that construction broke and the
            split point would be ambiguous or missing.
    """
    occurrences = sentence.count(value)
    if occurrences != 1:
        raise ValueError(
            f"expected value {value!r} to occur exactly once in sentence {sentence!r}, "
            f"found {occurrences} occurrence(s)"
        )
    split_at = sentence.index(value)
    return sentence[:split_at], value


def tier_a_prompts(person: Person, rng: np.random.Generator) -> dict[str, tuple[str, str]]:
    """One (prompt, answer) pair per attribute, drawn from the TRAIN template family.

    "Same phrasing family the model trained on" (blocking condition 4, Tier
    A): a fresh declarative-template draw per attribute, independent of
    whichever instance a given person's actual training row happened to use
    -- the family, not the literal instance, is what makes this Tier A rather
    than Tier B (see module docstring's Section 3 header).

    Args:
        person: individual to render.
        rng: caller-owned numpy Generator (never the global numpy state),
            matching render_training_text's own contract.

    Returns:
        Maps each of src.data.schema.ATTRIBUTES to a (prompt, answer) pair.
    """
    sentences = render_training_text(person, rng)
    return {
        attribute: _train_family_prompt_answer(sentence, person.attributes[attribute])
        for attribute, sentence in zip(ATTRIBUTES, sentences)
    }


def tier_b_prompts(person: Person, rng: np.random.Generator) -> dict[str, tuple[str, str]]:
    """One (prompt, answer) pair per attribute, drawn from the HELD-OUT PROBE template family.

    "Held-out probe phrasings" (blocking condition 4, Tier B): render_probe_qa
    draws from src.data.generate's PROBE template pool, which is structurally
    disjoint from the TRAIN pool (asserted at import time in
    src/data/generate.py). `template_idx` is drawn uniformly from
    [0, MIN_GUARANTEED_TEMPLATES_PER_ATTRIBUTE) per (person, attribute) --
    every attribute's probe pool is guaranteed to have at least this many
    entries (see that constant's provenance comment), so this never depends
    on the generator's exact, private pool sizes.

    Args:
        person: individual to render.
        rng: caller-owned numpy Generator controlling template selection.

    Returns:
        Maps each of src.data.schema.ATTRIBUTES to a (prompt, answer) pair.
    """
    result: dict[str, tuple[str, str]] = {}
    for attribute in ATTRIBUTES:
        template_idx = int(rng.integers(0, MIN_GUARANTEED_TEMPLATES_PER_ATTRIBUTE))
        result[attribute] = render_probe_qa(person, attribute, template_idx)
    return result


@dataclass(frozen=True, slots=True)
class TierResult:
    """One tier's aggregate evaluation over a set of people.

    Attributes:
        bits_recovered: sum over attributes of bits_recovered_per_attribute.
        entropy_ceiling: dataset_entropy_bits(len(people)) -- the exact upper
            bound bits_recovered cannot exceed (enforced per-item by
            src.metrics.bits.bits_recovered_per_attribute; this field is the
            reference value for reporting, not a second enforcement path).
        per_attribute_bits: bits recovered per attribute name.
    """

    bits_recovered: float
    entropy_ceiling: float
    per_attribute_bits: dict[str, float]


def _evaluate_tier(
    model: TinyLM,
    tokenizer: Tokenizer,
    people: Sequence[Person],
    prompts_fn: Callable[[Person, np.random.Generator], dict[str, tuple[str, str]]],
    rng: np.random.Generator,
) -> TierResult:
    """Score `model` on one (prompt, answer) pair per (person, attribute).

    Wires generate -> tokenize -> model -> bits: tokenizer.encode_qa builds
    the (ids, answer_mask) pair, the model produces logits,
    src.models.tiny_lm.causal_lm_shift aligns them to the
    src.metrics.bits.answer_cross_entropy_bits contract, and
    bits_recovered_per_attribute aggregates -- the same wiring
    tests/lead/lead_p0_control.py verifies independently (by recomputing
    cross-entropy from log-softmax rather than calling bits.py), but through
    the production components here rather than that file's from-scratch
    reimplementation.

    Args:
        model: a TinyLM in eval mode.
        tokenizer: the shared closed-vocabulary tokenizer.
        people: individuals to score.
        prompts_fn: tier_a_prompts or tier_b_prompts.
        rng: caller-owned numpy Generator controlling template selection.

    Returns:
        Aggregated bits recovered, entropy ceiling, and per-attribute
        breakdown over every (person, attribute) pair.

    Raises:
        ValueError: if `people` is empty.
        EntropyBoundViolation: propagated from
            src.metrics.bits.bits_recovered_per_attribute if any single
            item's cross-entropy implies recovering more bits than that
            item's entropy ceiling -- the project's highest-value assertion.
    """
    if not people:
        raise ValueError("people must be non-empty; nothing to evaluate")

    ce_bits_by_attribute: dict[str, list[float]] = {attribute: [] for attribute in ATTRIBUTES}
    for person in people:
        prompts = prompts_fn(person, rng)
        for attribute, (prompt, answer) in prompts.items():
            ids, mask = tokenizer.encode_qa(prompt, answer)
            input_ids = torch.tensor([ids], dtype=torch.long)
            answer_mask = torch.tensor([mask], dtype=torch.bool)
            with torch.no_grad():
                logits = model(input_ids)
            shifted_logits, targets, shifted_mask = causal_lm_shift(logits, input_ids, answer_mask)
            ce_bits_by_attribute[attribute].append(
                answer_cross_entropy_bits(shifted_logits, targets, shifted_mask)
            )

    per_attribute_bits = bits_recovered_per_attribute(ce_bits_by_attribute)
    return TierResult(
        bits_recovered=sum(per_attribute_bits.values()),
        entropy_ceiling=dataset_entropy_bits(len(people)),
        per_attribute_bits=per_attribute_bits,
    )


def _build_training_corpus(
    people: Sequence[Person],
    tokenizer: Tokenizer,
    block_size: int,
    template_rng: np.random.Generator,
    pad_id: int,
) -> torch.Tensor:
    """Tokenized, padded biography rows: one per person, ready for src.train.train_loop.

    Each row is the person's declarative biography (render_training_text,
    joined with a single space -- one sentence per attribute, in
    src.data.schema.ATTRIBUTES order), tokenized and right-padded to
    `block_size` with `pad_id`. `pad_id` never occurs in real generated text
    (verified: no attribute value or template word is the literal string
    "<pad>"), so padding cannot be confused with content.

    Never silently truncates (03_edge_cases_and_scares.md, Training/Huge): a
    biography that does not fit `block_size` raises, naming the person and
    the shortfall, rather than dropping tokens.

    Args:
        people: individuals to render, in row order.
        tokenizer: the shared closed-vocabulary tokenizer.
        block_size: fixed row length; must accommodate the longest generated
            biography (see DEFAULT_BLOCK_SIZE's provenance comment).
        template_rng: caller-owned numpy Generator controlling which TRAIN
            template each sentence uses (render_training_text's own
            contract).
        pad_id: token id for src.data.tokenizer.PAD.

    Returns:
        Token ids, shape (len(people), block_size), dtype torch.long.

    Raises:
        ValueError: if `people` is empty, or if any biography encodes to more
            than `block_size` tokens.
    """
    if not people:
        raise ValueError("people must be non-empty; cannot build a training corpus of size 0")

    rows: list[list[int]] = []
    for person in people:
        sentences = render_training_text(person, template_rng)
        ids = tokenizer.encode(" ".join(sentences))
        if len(ids) > block_size:
            raise ValueError(
                f"person {person.person_id!r}: biography encodes to {len(ids)} tokens, "
                f"exceeds block_size={block_size} by {len(ids) - block_size}; increase "
                f"--block-size rather than truncating"
            )
        rows.append(ids + [pad_id] * (block_size - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


# =============================================================================
# SECTION 4 -- ORCHESTRATION: one sweep point
# =============================================================================


@dataclass(frozen=True, slots=True)
class SweepPointResult:
    """Everything measured and recorded at one dataset size N.

    Every field the P1 brief requires recorded "in every result" (exposures,
    dtype, seed, N) is a top-level field here, not merely implied by the
    sweep-level config -- a single result dict is self-describing.
    """

    n_people: int
    exposures: int
    training_steps: int
    seed: int
    dtype: str
    parameter_count: int
    substrate_bytes: int
    batch_size: int
    lr: float
    final_train_loss_nats: float
    wall_clock_seconds: float
    tier_a: TierResult
    tier_b: TierResult
    gap_a_minus_b_bits: float


def run_sweep_point(
    *,
    n_people: int,
    seed: int,
    exposures: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    config: ModelConfig,
    tokenizer: Tokenizer,
    pad_id: int,
    train_template_seed: int,
    tier_a_template_seed: int,
    tier_b_template_seed: int,
    num_threads: int,
) -> SweepPointResult:
    """Train and evaluate one point of the P1 capacity sweep: dataset size `n_people`.

    CONTRACT
        in: n_people (>= batch_size), seed, exposures (>= MIN_EXPOSURES), a
            FIXED ModelConfig shared across every point in the caller's
            sweep (parameter_count must not vary with N -- that is the whole
            premise of a capacity sweep), and three independent template-
            selection seeds so template choice is reproducible regardless of
            which other N's have run in the same process.
        out: a SweepPointResult with both tiers' bits recovered, the plateau
            denominators (parameter_count, substrate_bytes), and enough
            provenance (exposures, dtype, seed) to interpret it standalone.
        errors: ValueError (n_people < batch_size, or propagated from
            _build_training_corpus / training_steps_for_exposures / train_loop);
            EntropyBoundViolation propagated from _evaluate_tier.
        invariants: model runs in TRAINING_DTYPE throughout;
            count_parameters(model) == analytic_param_count(config) (S3);
            both tiers score the SAME `people` (only phrasing differs, not
            personhood -- blocking condition 4).

    Non-goals (explicit, not silent): resumability is at whole-sweep-point
    granularity (Section 5), not mid-training-step granularity -- train_loop
    supports checkpoint/resume via out_dir/resume_path, but wiring that in
    here would add real complexity for uncertain benefit at container scale,
    where one point's training takes seconds, not hours.

    Raises:
        ValueError: if n_people < batch_size (src.train.ToyBatcher's own
            requirement, surfaced here with a clearer message naming the
            actual N), or propagated as described above.
    """
    if n_people < batch_size:
        raise ValueError(
            f"n_people={n_people} is smaller than batch_size={batch_size}; "
            f"src.train.ToyBatcher requires batch_size <= number of examples"
        )

    start = time.monotonic()

    people = list(generate_people(n_people, seed))

    train_rng = np.random.default_rng(train_template_seed)
    data = _build_training_corpus(people, tokenizer, config.block_size, train_rng, pad_id)

    steps = training_steps_for_exposures(exposures, n_train_examples=n_people, batch_size=batch_size)

    train_result = train_loop(
        config=config,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        data=data,
        num_threads=num_threads,
        weight_decay=weight_decay,
    )
    model = train_result.model
    if model is None:  # pragma: no cover -- train_loop always returns a model; defensive.
        raise RuntimeError("train_loop returned no model")
    model.eval()

    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != TRAINING_DTYPE:
        raise RuntimeError(
            f"model parameters are {actual_dtype}, expected {TRAINING_DTYPE} "
            f"(blocking condition 2: fp32 only in this container)"
        )

    tier_a = _evaluate_tier(
        model, tokenizer, people, tier_a_prompts, np.random.default_rng(tier_a_template_seed)
    )
    tier_b = _evaluate_tier(
        model, tokenizer, people, tier_b_prompts, np.random.default_rng(tier_b_template_seed)
    )

    parameter_count = count_parameters(model, include_embeddings=True)
    expected_parameter_count = analytic_param_count(config)
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"count_parameters(model)={parameter_count} != analytic_param_count(config)="
            f"{expected_parameter_count} (03_edge_cases_and_scares.md S3: a parameter-count "
            f"mismatch silently manufactures a capacity difference from nothing)"
        )
    substrate_bytes_value = substrate_bytes(parameter_count)

    return SweepPointResult(
        n_people=n_people,
        exposures=exposures,
        training_steps=steps,
        seed=seed,
        dtype=str(TRAINING_DTYPE),
        parameter_count=parameter_count,
        substrate_bytes=substrate_bytes_value,
        batch_size=batch_size,
        lr=lr,
        final_train_loss_nats=train_result.loss_history[-1] if train_result.loss_history else float("nan"),
        wall_clock_seconds=time.monotonic() - start,
        tier_a=tier_a,
        tier_b=tier_b,
        gap_a_minus_b_bits=tier_a.bits_recovered - tier_b.bits_recovered,
    )


# =============================================================================
# SECTION 5 -- SWEEP + RESUME
# =============================================================================


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Sweep-level configuration: identical across every point, recorded once
    and re-validated on every --resume so a resumed sweep cannot silently mix
    incompatible runs."""

    seed: int
    exposures: int
    batch_size: int
    lr: float
    weight_decay: float
    n_layer: int
    n_head: int
    d_model: int
    block_size: int
    vocab_size: int
    num_threads: int
    sizes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sizes"] = list(self.sizes)
        payload["dtype"] = str(TRAINING_DTYPE)
        return payload


def _tier_result_to_dict(tier: TierResult) -> dict[str, Any]:
    return {
        "bits_recovered": tier.bits_recovered,
        "entropy_ceiling": tier.entropy_ceiling,
        "per_attribute_bits": tier.per_attribute_bits,
    }


def _tier_result_from_dict(payload: dict[str, Any]) -> TierResult:
    return TierResult(
        bits_recovered=payload["bits_recovered"],
        entropy_ceiling=payload["entropy_ceiling"],
        per_attribute_bits=payload["per_attribute_bits"],
    )


def _point_to_dict(point: SweepPointResult) -> dict[str, Any]:
    payload = asdict(point)
    payload["tier_a"] = _tier_result_to_dict(point.tier_a)
    payload["tier_b"] = _tier_result_to_dict(point.tier_b)
    return payload


def _point_from_dict(payload: dict[str, Any]) -> SweepPointResult:
    payload = dict(payload)
    payload["tier_a"] = _tier_result_from_dict(payload["tier_a"])
    payload["tier_b"] = _tier_result_from_dict(payload["tier_b"])
    return SweepPointResult(**payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` atomically (tmp file + rename).

    Mirrors src.train.save_checkpoint's tmp-then-replace pattern: a process
    killed mid-write (03_edge_cases_and_scares.md, Training/Failure) leaves
    either the previous complete file or a complete new one, never a
    half-written one -- required for resumability (implementation/01_phases_and_gates.md,
    "Quota and interruption policy").
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _tier_capacity_report(
    n_values: list[int],
    bits_values: list[float],
    parameter_count: int,
    substrate_bytes_value: int,
    threshold: float,
) -> dict[str, Any]:
    """One tier's saturation verdict plus capacity numbers, or the reason they are withheld.

    detect_saturation raises ValueError on several malformed-or-inconclusive
    curve shapes (see its Raises section) -- correct for a function whose
    direct callers (bits_per_parameter_at_plateau, tests) must not silently
    misjudge saturation. A sweep-level report has a different job: it must
    always produce SOMETHING for a human to read, including "cannot tell
    yet" (a real, expected state at low exposures, where the smallest N's may
    not have trained long enough to move off the chance-level clamp at all --
    observed directly while building this file: two flat-zero points followed
    by a rising one). ValueError is therefore caught here, once, and its
    message preserved verbatim in the report rather than hidden -- the
    underlying cause is never lost, only kept from crashing the whole sweep
    summary.
    """
    try:
        verdict = detect_saturation(n_values, bits_values, threshold=threshold)
    except ValueError as exc:
        return {
            "is_saturated": False,
            "decay_ratio": None,
            "remaining_fraction": None,
            "reason": None,
            "cannot_judge_saturation_reason": str(exc),
            "plateau_bits_recovered": bits_values[-1],
            "bits_per_parameter": None,
            "bits_per_substrate_byte": None,
        }

    report: dict[str, Any] = {
        "is_saturated": verdict.is_saturated,
        "decay_ratio": verdict.decay_ratio,
        "remaining_fraction": verdict.remaining_fraction,
        "reason": verdict.reason,
        "plateau_bits_recovered": bits_values[-1],
        "bits_per_parameter": None,
        "bits_per_substrate_byte": None,
    }
    if verdict.is_saturated:
        report["bits_per_parameter"] = bits_per_parameter_at_plateau(
            n_values, bits_values, parameter_count, threshold=threshold
        )
        report["bits_per_substrate_byte"] = bits_per_substrate_byte_at_plateau(
            n_values, bits_values, substrate_bytes_value, threshold=threshold
        )
    else:
        report["capacity_not_reported_reason"] = (
            "curve is not saturated; a capacity number here would measure dataset size N, "
            "not model capacity (03_edge_cases_and_scares.md S4)"
        )
    return report


def build_report(points: list[SweepPointResult], threshold: float = MAX_REMAINING_FRACTION) -> dict[str, Any]:
    """Saturation verdicts and (if saturated) capacity numbers for both tiers.

    Args:
        points: sweep points measured so far, any order, any subset of the
            full --sizes list (a partial/resumed sweep is a legitimate
            input).
        threshold: forwarded to detect_saturation.

    Returns:
        A dict with "tier_a", "tier_b" (each from _tier_capacity_report, or a
        single "insufficient_points" note if fewer than
        MIN_SWEEP_POINTS_FOR_SATURATION points exist yet), and, when both
        tiers have enough points, "gap_a_minus_b_bits_at_largest_n".
    """
    sorted_points = sorted(points, key=lambda p: p.n_people)
    if len(sorted_points) < MIN_SWEEP_POINTS_FOR_SATURATION:
        note = (
            f"only {len(sorted_points)} point(s) completed; need >= "
            f"{MIN_SWEEP_POINTS_FOR_SATURATION} to judge saturation"
        )
        return {"tier_a": {"insufficient_points": note}, "tier_b": {"insufficient_points": note}}

    n_values = [p.n_people for p in sorted_points]
    last = sorted_points[-1]
    tier_a_bits = [p.tier_a.bits_recovered for p in sorted_points]
    tier_b_bits = [p.tier_b.bits_recovered for p in sorted_points]
    return {
        "tier_a": _tier_capacity_report(
            n_values, tier_a_bits, last.parameter_count, last.substrate_bytes, threshold
        ),
        "tier_b": _tier_capacity_report(
            n_values, tier_b_bits, last.parameter_count, last.substrate_bytes, threshold
        ),
        "gap_a_minus_b_bits_at_largest_n": last.gap_a_minus_b_bits,
    }


def _derive_template_seeds(seed: int) -> tuple[int, int, int]:
    """Three mutually independent seeds for train/tier-A/tier-B template selection.

    Spawned from a single numpy.random.SeedSequence(seed) in one call, which
    is what makes them mutually independent (numpy.random.SeedSequence.spawn's
    documented guarantee; a fresh SeedSequence(seed) constructed separately
    per stream would NOT be independent -- see src/data/generate.py's
    _derive_name_key_and_attr_seeds docstring for the same point, verified
    there empirically). These seeds are reused, unchanged, across every N in
    one sweep, so template choice for a given person is identical regardless
    of which other N's share the process (generate_people's own determinism
    guarantee: person index i is the same Person for any N > i under the same
    seed).
    """
    root = np.random.SeedSequence(seed)
    children = root.spawn(3)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)  # type: ignore[return-value]


def run_sweep(
    *,
    sizes: Sequence[int],
    seed: int,
    exposures: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    n_layer: int,
    n_head: int,
    d_model: int,
    block_size: int,
    num_threads: int,
    out_path: Path,
    resume: bool,
    saturation_threshold: float = MAX_REMAINING_FRACTION,
    on_point_done: Callable[[SweepPointResult], None] | None = None,
) -> dict[str, Any]:
    """Drive run_sweep_point across `sizes`, resumably, writing `out_path` after every point.

    Resumability (implementation/01_phases_and_gates.md, "Quota and
    interruption policy"): if `out_path` exists, `resume=False` refuses to
    overwrite it (FileExistsError, matching src.train's
    _guard_fresh_out_dir -- "two runs, same output dir -> fail fast, do not
    interleave", 03_edge_cases_and_scares.md Training/Concurrent).
    `resume=True` loads it, validates its SweepConfig matches this call's
    config exactly (a mismatch means "not the same sweep", raised rather than
    silently mixed), and trains only the N's not already present.

    Args:
        sizes: dataset sizes to sweep, need not be sorted or unique (this
            function sorts and dedupes).
        seed, exposures, batch_size, lr, weight_decay: forwarded to every
            run_sweep_point call, and to SweepConfig for resume validation.
        n_layer, n_head, d_model, block_size: fixed ModelConfig for the whole
            sweep (vocab_size comes from the shared tokenizer).
        num_threads: pinned torch thread count (H3).
        out_path: results file. Written after every completed point.
        resume: see above.
        saturation_threshold: forwarded to build_report.
        on_point_done: optional callback invoked with each SweepPointResult
            as it completes (used by the CLI for progress printing; tests can
            observe progress without parsing stdout).

    Returns:
        {"sweep_config": ..., "points": [...], "report": ...} -- the same
        payload written to `out_path`.

    Raises:
        FileExistsError: if `out_path` exists and `resume` is False.
        FileNotFoundError: if `resume` is True and `out_path` does not exist.
        ValueError: if a resumed sweep's stored config does not match this
            call's config, or if `sizes` is empty or contains a non-positive
            value, or propagated from run_sweep_point.
    """
    unique_sizes = sorted(set(sizes))
    if not unique_sizes:
        raise ValueError("sizes must name at least one N")
    if any(n <= 0 for n in unique_sizes):
        raise ValueError(f"sizes must all be positive, got {unique_sizes}")

    run_self_checks()

    tokenizer = build_tokenizer()
    pad_id = tokenizer.encode(PAD)[0]
    config = ModelConfig(
        n_layer=n_layer, n_head=n_head, d_model=d_model,
        vocab_size=tokenizer.vocab_size, block_size=block_size, tie_embeddings=True,
    )
    sweep_config = SweepConfig(
        seed=seed, exposures=exposures, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        n_layer=n_layer, n_head=n_head, d_model=d_model, block_size=block_size,
        vocab_size=tokenizer.vocab_size, num_threads=num_threads, sizes=tuple(unique_sizes),
    )

    points: list[SweepPointResult] = []
    if out_path.exists():
        if not resume:
            raise FileExistsError(
                f"{out_path} already exists and resume=False; refusing to overwrite or "
                f"interleave with a previous sweep. Pass resume=True to continue it."
            )
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing["sweep_config"] != sweep_config.to_dict():
            raise ValueError(
                f"cannot resume: {out_path}'s sweep_config {existing['sweep_config']} != "
                f"requested config {sweep_config.to_dict()}"
            )
        points = [_point_from_dict(p) for p in existing["points"]]
    elif resume:
        raise FileNotFoundError(f"resume=True but {out_path} does not exist")

    train_template_seed, tier_a_template_seed, tier_b_template_seed = _derive_template_seeds(seed)

    done_sizes = {p.n_people for p in points}
    for n in unique_sizes:
        if n in done_sizes:
            continue
        point = run_sweep_point(
            n_people=n, seed=seed, exposures=exposures, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, config=config, tokenizer=tokenizer, pad_id=pad_id,
            train_template_seed=train_template_seed, tier_a_template_seed=tier_a_template_seed,
            tier_b_template_seed=tier_b_template_seed, num_threads=num_threads,
        )
        points.append(point)
        points.sort(key=lambda p: p.n_people)
        _atomic_write_json(out_path, {
            "sweep_config": sweep_config.to_dict(),
            "points": [_point_to_dict(p) for p in points],
        })
        if on_point_done is not None:
            on_point_done(point)

    report = build_report(points, saturation_threshold)
    payload = {
        "sweep_config": sweep_config.to_dict(),
        "points": [_point_to_dict(p) for p in points],
        "report": report,
    }
    _atomic_write_json(out_path, payload)
    return payload


# =============================================================================
# SECTION 6 -- SELF-CHECKS
# Assertions of identities this file depends on. Run before any sweep starts;
# a violated identity means nothing the sweep reports should be believed.
# =============================================================================


def run_self_checks() -> None:
    """Assert this file's structural assumptions about USE-ONLY dependencies.

    Raises:
        AssertionError: if a depended-upon invariant of src/data/* breaks.
    """
    sample_person = next(iter(generate_people(1, seed=0)))
    for attribute in ATTRIBUTES:
        for template_idx in range(MIN_GUARANTEED_TEMPLATES_PER_ATTRIBUTE):
            # Raises ValueError if the probe pool for this attribute turns out
            # to be smaller than MIN_GUARANTEED_TEMPLATES_PER_ATTRIBUTE --
            # this constant's provenance comment claims that cannot happen,
            # so this check turns a silent, wrong assumption into a loud one.
            render_probe_qa(sample_person, attribute, template_idx)

    tokenizer = build_tokenizer()
    pad_ids = tokenizer.encode(PAD)
    assert len(pad_ids) == 1, f"PAD must encode to exactly one token, got {pad_ids}"


# =============================================================================
# SECTION 7 -- CLI / REPORT
# Formatting and argument parsing only. No arithmetic beyond calling
# Sections 2, 4, 5.
# =============================================================================


def _parse_sizes(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated --sizes argument into a sorted tuple of positive, distinct ints.

    Raises:
        ValueError: if `raw` is not comma-separated ints, is empty, contains
            a non-positive value, or contains a duplicate.
    """
    try:
        sizes = tuple(int(part) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"--sizes must be comma-separated ints, got {raw!r}") from exc
    if len(sizes) == 0:
        raise ValueError("--sizes must name at least one N")
    if any(n <= 0 for n in sizes):
        raise ValueError(f"--sizes must all be positive, got {sizes}")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"--sizes contains duplicates: {sizes}")
    return tuple(sorted(sizes))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P1 capacity-sweep harness: bits_recovered vs dataset size N, two "
            "evaluation tiers (A: same phrasing family; B: held-out phrasing), "
            "gated by an explicit saturation check. See "
            "implementation/01_phases_and_gates.md P1. CONTAINER-ONLY unless run "
            "on the target Mac mini."
        )
    )
    parser.add_argument("--seed", type=int, required=True, help="determinism seed")
    parser.add_argument(
        "--exposures", type=int, required=True,
        help="target training exposures per fact (mistake M4); must be >= 1, recorded in every result",
    )
    parser.add_argument(
        "--sizes", type=str, required=True,
        help="comma-separated dataset sizes N to sweep, e.g. 16,32,64,128",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help="max tokens per training row; raises (never truncates) if a biography exceeds it",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="pinned torch thread count (H3); must match across runs being compared",
    )
    parser.add_argument("--saturation-threshold", type=float, default=MAX_REMAINING_FRACTION)
    parser.add_argument("--out", type=Path, required=True, help="results JSON path")
    parser.add_argument("--resume", action="store_true", help="continue a previous run at --out")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 success, 1 on a raised error)."""
    args = _build_arg_parser().parse_args(argv)
    try:
        sizes = _parse_sizes(args.sizes)

        def _print_progress(point: SweepPointResult) -> None:
            print(
                f"N={point.n_people}: tier_a={point.tier_a.bits_recovered:.2f} bits, "
                f"tier_b={point.tier_b.bits_recovered:.2f} bits, "
                f"gap={point.gap_a_minus_b_bits:.2f} bits, steps={point.training_steps}, "
                f"wall_clock={point.wall_clock_seconds:.1f}s"
            )

        payload = run_sweep(
            sizes=sizes, seed=args.seed, exposures=args.exposures, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, n_layer=args.n_layer, n_head=args.n_head,
            d_model=args.d_model, block_size=args.block_size, num_threads=args.num_threads,
            out_path=args.out, resume=args.resume, saturation_threshold=args.saturation_threshold,
            on_point_done=_print_progress,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"CAPACITY SWEEP FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
