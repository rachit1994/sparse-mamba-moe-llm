#!/usr/bin/env python3
"""Lead-owned independent plateau detector, plus a known-answer battery.

WHY THIS EXISTS
    The plateau detector is the single guard preventing the project from
    reporting a capacity number computed on an unsaturated curve. If a rising
    curve is mistaken for a plateau, `bits / params` measures DATASET SIZE
    rather than model capacity (edge case S4) and the headline number is wrong
    with nothing raised.

    Written independently of src/capacity_sweep.py so agreement between the two
    is evidence rather than a shared bug.

CRITERION, AND WHY THE FIRST ONE WAS WRONG
    First attempt thresholded the ratio of final to first marginal gain at 15%.
    That threshold was arbitrary: it had no stated relationship to the quantity
    anyone cares about, which is the ERROR IN THE CAPACITY ESTIMATE.

    Running it exposed the problem. An exponential at 91% of its ceiling scored
    0.165 and was rejected, while a case labelled "textbook plateau" turned out
    on inspection not to have plateaued at all -- reading capacity there would
    have understated it by ~9%.

    The fix is not to move the threshold until the cases pass. It is to define
    saturation by the thing that matters:

        remaining_fraction = (extrapolated asymptote - y_last) / y_last

    Marginal gains of a saturating curve decay geometrically. Estimate that
    decay from the observed gains, sum the remaining geometric series, and call
    the curve saturated when the unmeasured remainder is under 5% of the value
    read. The threshold is then a statement about tolerable capacity error, not
    a magic number.

HOW TO REVIEW
    Section 1  CONSTANTS  - the tolerance, expressed as capacity error.
    Section 2  DETECTOR   - pure functions, one quantity each.
    Section 3  BATTERY    - curves whose correct answer follows from arithmetic
                            stated in the case itself, not from intuition.
    Section 4  REPORT     - formatting only.

Run: python3 tests/lead/lead_plateau_check.py     (exit 0 = battery passed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# =============================================================================
# SECTION 1 -- CONSTANTS
# =============================================================================

#: Largest tolerable underestimate of capacity, as a fraction of the value read
#: off the curve. At 5%, a capacity reported as 2.00 bits/param could truly be
#: up to 2.10 -- comfortably inside P1's +/-25% gate, so saturation is not the
#: dominant error term.
MAX_REMAINING_FRACTION = 0.05

#: Fewest points needed. Estimating the decay of marginal gains needs at least
#: three gains, hence four points.
MIN_POINTS_FOR_JUDGEMENT = 4

#: If observed gains do not decay (ratio >= 1), the curve is still linear and no
#: finite asymptote can be extrapolated.
NO_DECAY_RATIO = 1.0


# =============================================================================
# SECTION 2 -- DETECTOR
# =============================================================================


@dataclass(frozen=True)
class PlateauVerdict:
    """Result of a saturation test.

    Attributes:
        is_saturated: whether capacity may be read off this curve.
        decay_ratio: geometric decay factor of marginal gains, per interval.
        remaining_fraction: extrapolated unmeasured gain, as a fraction of the
            last measured value. This is the capacity underestimate.
        reason: human-readable justification for the report.
    """

    is_saturated: bool
    decay_ratio: float
    remaining_fraction: float
    reason: str


def _marginal_gains(x_values: list[float], y_values: list[float]) -> list[float]:
    """Per-unit-x gain over each consecutive interval."""
    return [
        (y1 - y0) / (x1 - x0)
        for x0, x1, y0, y1 in zip(x_values, x_values[1:], y_values, y_values[1:])
    ]


def _estimate_decay_ratio(gains: list[float]) -> float:
    """Geometric decay factor of marginal gains, estimated over the whole curve.

    Uses the endpoints of the gain sequence rather than an average of noisy
    consecutive ratios: a single small gain in the middle would otherwise
    dominate. Returns NO_DECAY_RATIO when gains do not shrink.
    """
    first, last = gains[0], gains[-1]
    if first <= 0 or last <= 0:
        return NO_DECAY_RATIO
    n_steps = len(gains) - 1
    if n_steps < 1:
        return NO_DECAY_RATIO
    return (last / first) ** (1.0 / n_steps)


def detect_plateau(
    x_values: list[float],
    y_values: list[float],
    max_remaining: float = MAX_REMAINING_FRACTION,
) -> PlateauVerdict:
    """Decide whether capacity may be read off this curve.

    Extrapolates the remaining gain as a geometric series in the observed decay
    of marginal gains, and saturates when that remainder is a small fraction of
    the value already measured.

    Args:
        x_values: dataset sizes, strictly increasing.
        y_values: bits recovered at each size.
        max_remaining: tolerable capacity underestimate, as a fraction.

    Returns:
        A PlateauVerdict. Callers MUST refuse to emit a capacity number when
        `is_saturated` is False.

    Raises:
        ValueError: on mismatched lengths, too few points, non-increasing x, a
            non-positive final value, or a non-rising start (for which the
            extrapolation is undefined).
    """
    if len(x_values) != len(y_values):
        raise ValueError(
            f"x and y must be the same length, got {len(x_values)} and {len(y_values)}"
        )
    if len(x_values) < MIN_POINTS_FOR_JUDGEMENT:
        raise ValueError(
            f"need >= {MIN_POINTS_FOR_JUDGEMENT} points, got {len(x_values)}"
        )
    if any(b <= a for a, b in zip(x_values, x_values[1:])):
        raise ValueError("x_values must be strictly increasing")
    if y_values[-1] <= 0:
        raise ValueError(f"final y must be positive, got {y_values[-1]}")

    gains = _marginal_gains(x_values, y_values)
    if gains[0] <= 0:
        raise ValueError(
            f"first-interval gain must be positive, got {gains[0]}; a curve that "
            f"does not rise at the start cannot be extrapolated this way"
        )

    decay = _estimate_decay_ratio(gains)
    if decay >= NO_DECAY_RATIO:
        return PlateauVerdict(
            is_saturated=False,
            decay_ratio=decay,
            remaining_fraction=math.inf,
            reason="marginal gains are not decaying; no finite asymptote to read",
        )

    # Remaining gain, summing the geometric tail beyond the last measured point.
    last_step = x_values[-1] - x_values[-2]
    remaining_absolute = gains[-1] * last_step * decay / (1.0 - decay)
    remaining_fraction = remaining_absolute / y_values[-1]

    saturated = remaining_fraction < max_remaining
    reason = (
        f"gains decay {decay:.3f}x per interval; extrapolated remaining gain is "
        f"{remaining_fraction:.1%} of the measured value "
        f"({'within' if saturated else 'over'} the {max_remaining:.0%} tolerance)"
    )
    return PlateauVerdict(saturated, decay, remaining_fraction, reason)


# =============================================================================
# SECTION 3 -- KNOWN-ANSWER BATTERY
# Each case states the arithmetic that fixes its expected answer, so the
# expectation is derived rather than guessed. The first version of this file
# guessed, and was wrong.
# =============================================================================


@dataclass(frozen=True)
class BatteryCase:
    name: str
    x_values: list[float]
    y_values: list[float]
    expected_saturated: bool
    why: str


def _linear(n_points: int = 8, slope: float = 100.0) -> tuple[list[float], list[float]]:
    xs = [float(i + 1) * 1000 for i in range(n_points)]
    return xs, [slope * x for x in xs]


def _saturating(
    n_points: int, ceiling: float, rate: float
) -> tuple[list[float], list[float]]:
    xs = [float(i + 1) * 1000 for i in range(n_points)]
    return xs, [ceiling * (1.0 - math.exp(-rate * x)) for x in xs]


def _fraction_of_ceiling(rate: float, x_last: float) -> float:
    """Exact fraction of the asymptote reached, for an exponential curve."""
    return 1.0 - math.exp(-rate * x_last)


def _noisy(values: list[float], amplitude: float, seed: int = 7) -> list[float]:
    """Deterministic pseudo-noise so the battery stays reproducible."""
    out, state = [], seed
    for value in values:
        state = (1103515245 * state + 12345) % (2**31)
        out.append(value * (1.0 + (state / (2**31) - 0.5) * 2 * amplitude))
    return out


def build_battery() -> tuple[BatteryCase, ...]:
    """Curves whose expected verdict follows from stated arithmetic."""
    lin_x, lin_y = _linear()

    # Reached 1 - e^(-3e-4 * 8000) = 90.9% of ceiling => ~10% still missing,
    # which exceeds the 5% tolerance. NOT saturated. The first version of this
    # file called this a "textbook plateau"; the arithmetic says otherwise.
    partial_x, partial_y = _saturating(8, 1e6, 3e-4)

    # Reached 1 - e^(-1e-3 * 8000) = 99.97% of ceiling => ~0.03% missing.
    full_x, full_y = _saturating(8, 1e6, 1e-3)

    # Reached 1 - e^(-5e-5 * 8000) = 33% of ceiling => far from saturated.
    early_x, early_y = _saturating(8, 1e6, 5e-5)

    return (
        BatteryCase(
            "straight line", lin_x, lin_y, False,
            "gains never decay; no asymptote exists. Accepting this is the S4 failure",
        ),
        BatteryCase(
            "exponential, 99.97% of ceiling", full_x, full_y, True,
            f"reached {_fraction_of_ceiling(1e-3, 8000):.4%}; remainder well under 5%",
        ),
        BatteryCase(
            "exponential, 90.9% of ceiling", partial_x, partial_y, False,
            f"reached {_fraction_of_ceiling(3e-4, 8000):.2%}; ~9% missing exceeds 5% tolerance",
        ),
        BatteryCase(
            "exponential, 33% of ceiling", early_x, early_y, False,
            f"reached {_fraction_of_ceiling(5e-5, 8000):.2%}; nowhere near an asymptote",
        ),
        BatteryCase(
            "noisy straight line", lin_x, _noisy(lin_y, 0.05), False,
            "noise must not manufacture a false plateau",
        ),
        BatteryCase(
            "noisy 99.97% exponential", full_x, _noisy(full_y, 0.02), True,
            "noise must not hide a real plateau",
        ),
    )


def run_battery() -> list[tuple[BatteryCase, PlateauVerdict, bool]]:
    results = []
    for case in build_battery():
        verdict = detect_plateau(case.x_values, case.y_values)
        results.append((case, verdict, verdict.is_saturated == case.expected_saturated))
    return results


def check_error_handling() -> list[tuple[str, bool]]:
    """Malformed input must raise rather than return a plausible answer."""

    def raises(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    return [
        ("too few points", raises(lambda: detect_plateau([1, 2], [1, 2]))),
        ("length mismatch", raises(lambda: detect_plateau([1, 2, 3, 4], [1, 2, 3]))),
        ("non-increasing x", raises(lambda: detect_plateau([1, 2, 2, 4], [1, 2, 3, 4]))),
        ("flat start", raises(lambda: detect_plateau([1, 2, 3, 4], [5, 5, 5, 5]))),
        ("negative final y", raises(lambda: detect_plateau([1, 2, 3, 4], [1, 2, 3, -1]))),
    ]


# =============================================================================
# SECTION 4 -- REPORT
# =============================================================================


def main() -> int:
    print("=" * 78)
    print("LEAD INDEPENDENT PLATEAU DETECTOR -- known-answer battery")
    print("=" * 78)
    print(f"criterion: extrapolated remaining gain < {MAX_REMAINING_FRACTION:.0%} "
          f"of the measured value\n")

    print(f"{'curve':<34}{'decay':>8}{'remaining':>11}{'verdict':>9}{'want':>7}{'':>6}")
    all_passed = True
    for case, verdict, passed in run_battery():
        all_passed &= passed
        remaining = (
            "inf" if math.isinf(verdict.remaining_fraction)
            else f"{verdict.remaining_fraction:.2%}"
        )
        print(
            f"{case.name:<34}{verdict.decay_ratio:>8.3f}{remaining:>11}"
            f"{str(verdict.is_saturated):>9}{str(case.expected_saturated):>7}"
            f"{'  OK' if passed else '  FAIL':>6}"
        )
        if not passed:
            print(f"      why: {case.why}")

    print("\nmalformed input:")
    for name, raised in check_error_handling():
        all_passed &= raised
        print(f"  {name:<22}{'raises' if raised else '*** DID NOT RAISE ***'}")

    print()
    print("BATTERY:", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
