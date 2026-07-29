#!/usr/bin/env python3
"""Can P1 reach the 1000-exposure regime in this container, or does it need the Mac mini?

THE QUESTION
    P1 must reproduce Allen-Zhu's 2.0 bits/param, which requires each fact seen
    ~1000 times AND a dataset large enough to saturate the model. Those two
    requirements multiply. This file measures the container's actual training
    throughput and computes whether the product is reachable.

    Answering "no" is a legitimate finding. Faking a short run and reporting a
    capacity number would be mistake M4 repeating.

HOW TO REVIEW THIS FILE
    Section 1  CONSTANTS      - inputs with provenance.
    Section 2  PURE FUNCTIONS - one quantity each, no I/O.
    Section 3  MEASUREMENT    - the one empirical input, clearly isolated.
    Section 4  REPORT         - formatting only.

Run: python3 verify_p1_feasibility.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

# =============================================================================
# SECTION 1 -- CONSTANTS
# =============================================================================

SECONDS_PER_DAY = 86_400

#: Knowledge bits per parameter at fp32/fp16/int8.
#: Source: Allen-Zhu & Li, Physics of Language Models Part 3.3 (ICLR 2025).
KNOWLEDGE_BITS_PER_PARAM = 2.0

#: Exposures per fact required to reach the 2.0 figure. At ~100 exposures the
#: measured ceiling drops to 1.0 bits/param. Source: same paper.
REQUIRED_EXPOSURES = 1000

#: Entropy of one synthetic person under the flat generator.
#: Source: src/data/schema.py BITS_PER_PERSON.
BITS_PER_PERSON = 50.97

#: Attributes stated per person, hence training statements per person.
#: Source: src/data/schema.py ATTRIBUTE_CARDINALITY.
FACTS_PER_PERSON = 5

#: Tokens per training statement, measured from real generated text
#: (e.g. "Kavion Rothovic-4465 comes from Crelldale." with digit splitting).
TOKENS_PER_STATEMENT = 14

#: Training cost in FLOPs is ~6 * params * tokens (fwd 2, bwd 4).
#: Source: Kaplan et al. scaling-laws convention, standard.
FLOPS_PER_PARAM_PER_TOKEN = 6

#: To saturate, dataset entropy must EXCEED model capacity. Below 1.0 the model
#: memorises everything and the curve never plateaus (edge case S4).
SATURATION_MARGIN = 2.0

#: Fraction of peak matmul throughput that a real training loop achieves.
#:
#: measure_container_throughput() times a bare 512x512 matmul. An actual step
#: also pays for attention, layernorms, the optimiser, autograd bookkeeping, and
#: Python dispatch, none of which are dense matmul. For small models on CPU the
#: non-matmul overhead dominates.
#:
#: This factor is an ESTIMATE, not a measurement, and it is deliberately given as
#: a range so the report shows optimistic and pessimistic bounds rather than a
#: single misleadingly precise figure. Reporting the raw matmul number as if it
#: were training throughput would understate wall-clock by ~10x.
TRAINING_EFFICIENCY_OPTIMISTIC = 0.20
TRAINING_EFFICIENCY_PESSIMISTIC = 0.05


@dataclass(frozen=True)
class ModelSize:
    """A candidate model for the sweep."""

    name: str
    n_params: int


CANDIDATE_MODELS: tuple[ModelSize, ...] = (
    ModelSize("tiny", 27_008),
    ModelSize("100k", 100_000),
    ModelSize("1m", 1_057_152),
    ModelSize("10m", 10_844_160),
)


# =============================================================================
# SECTION 2 -- PURE FUNCTIONS
# =============================================================================


def model_capacity_bits(n_params: int) -> float:
    """Knowledge bits the model can hold at the Allen-Zhu rate."""
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    return KNOWLEDGE_BITS_PER_PARAM * n_params


def people_needed_to_saturate(n_params: int) -> int:
    """Dataset size whose entropy exceeds model capacity by SATURATION_MARGIN.

    Below this the curve cannot plateau, and any capacity number derived from it
    measures dataset size rather than model capacity.
    """
    required_bits = model_capacity_bits(n_params) * SATURATION_MARGIN
    return int(required_bits / BITS_PER_PERSON) + 1


def training_tokens(n_people: int, exposures: int) -> int:
    """Total tokens processed for `n_people` at `exposures` repetitions each."""
    if exposures < 1:
        raise ValueError(f"exposures must be >= 1, got {exposures}")
    statements = n_people * FACTS_PER_PERSON * exposures
    return statements * TOKENS_PER_STATEMENT


def training_flops(n_params: int, n_tokens: int) -> float:
    """FLOPs to train `n_params` over `n_tokens`."""
    return FLOPS_PER_PARAM_PER_TOKEN * n_params * n_tokens


def wall_clock_days(flops: float, flops_per_second: float) -> float:
    """Days of compute at a measured throughput."""
    if flops_per_second <= 0:
        raise ValueError(f"flops_per_second must be positive, got {flops_per_second}")
    return flops / flops_per_second / SECONDS_PER_DAY


# =============================================================================
# SECTION 3 -- MEASUREMENT
# The single empirical input. Isolated so a reviewer can see exactly what was
# measured and what was assumed.
# =============================================================================


def measure_container_throughput(matrix_dim: int = 512, iterations: int = 40) -> float:
    """Measure achievable dense-matmul FLOP/s in this container.

    Uses a matmul of the scale the sweep actually performs, not a large GEMM
    that would overstate throughput for small models.

    Returns:
        Achieved FLOP/s, measured. This is CONTAINER-ONLY and does not transfer
        to the M4 Mac mini.
    """
    torch.set_num_threads(4)
    left = torch.randn(matrix_dim, matrix_dim)
    right = torch.randn(matrix_dim, matrix_dim)
    for _ in range(5):  # warm up
        left @ right

    start = time.perf_counter()
    for _ in range(iterations):
        product = left @ right
    _ = product[0, 0].item()  # force materialisation
    elapsed = time.perf_counter() - start

    flops_per_matmul = 2 * matrix_dim**3
    return flops_per_matmul * iterations / elapsed


# =============================================================================
# SECTION 4 -- REPORT
# =============================================================================


def _heading(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def report_feasibility(flops_per_second: float) -> bool:
    """Print the feasibility table. Returns True if any model is reachable."""
    _heading("P1 FEASIBILITY IN THIS CONTAINER (CONTAINER-ONLY)")
    print(f"Peak matmul throughput (MEASURED): {flops_per_second / 1e9:.1f} GFLOP/s "
          f"(4 CPU threads, torch {torch.__version__})")
    print(f"Training efficiency (ESTIMATED)  : "
          f"{TRAINING_EFFICIENCY_PESSIMISTIC:.0%}-{TRAINING_EFFICIENCY_OPTIMISTIC:.0%} of peak")
    print(f"=> effective                     : "
          f"{flops_per_second * TRAINING_EFFICIENCY_PESSIMISTIC / 1e9:.1f}-"
          f"{flops_per_second * TRAINING_EFFICIENCY_OPTIMISTIC / 1e9:.1f} GFLOP/s")
    print(f"Requirement: {REQUIRED_EXPOSURES} exposures/fact AND a saturating dataset\n")

    print(f"{'model':<8}{'params':>12}{'N to saturate':>15}{'tokens':>12}"
          f"{'days (opt)':>12}{'days (pess)':>13}")
    any_reachable = False
    for model in CANDIDATE_MODELS:
        n_people = people_needed_to_saturate(model.n_params)
        tokens = training_tokens(n_people, REQUIRED_EXPOSURES)
        flops = training_flops(model.n_params, tokens)
        days_opt = wall_clock_days(flops, flops_per_second * TRAINING_EFFICIENCY_OPTIMISTIC)
        days_pess = wall_clock_days(flops, flops_per_second * TRAINING_EFFICIENCY_PESSIMISTIC)
        if days_pess <= 1.0:
            any_reachable = True
        print(f"{model.name:<8}{model.n_params:>12,}{n_people:>15,}"
              f"{tokens / 1e6:>9.1f} M{days_opt:>12.2f}{days_pess:>13.2f}")

    print("\nPlan against the PESSIMISTIC column. A sweep needs 5-6 N values, so")
    print("multiply again by that. Only rows under ~1 day pessimistic are usable.")
    return any_reachable


def report_reduced_exposure_options(flops_per_second: float) -> None:
    _heading("WHAT IS REACHABLE AT REDUCED EXPOSURES")
    print("At fewer exposures the measured ceiling falls (1000 -> 2.0 b/param,")
    print("100 -> 1.0 b/param), so these runs CANNOT be compared to the 2.0")
    print("anchor. They can still validate the sweep machinery.\n")

    model = CANDIDATE_MODELS[0]
    n_people = people_needed_to_saturate(model.n_params)
    print(f"Using the '{model.name}' model ({model.n_params:,} params, "
          f"N={n_people:,} to saturate):\n")
    print(f"{'exposures':>10}{'tokens':>13}{'minutes*':>12}{'valid for 2.0 anchor?':>24}")
    for exposures in (10, 50, 100, 1000):
        tokens = training_tokens(n_people, exposures)
        minutes = wall_clock_days(
            training_flops(model.n_params, tokens),
            flops_per_second * TRAINING_EFFICIENCY_PESSIMISTIC,
        ) * 24 * 60
        valid = "yes" if exposures >= REQUIRED_EXPOSURES else "NO - machinery only"
        print(f"{exposures:>10}{tokens / 1e6:>10.2f} M{minutes:>12.1f}{valid:>24}")


def main() -> int:
    flops_per_second = measure_container_throughput()
    reachable = report_feasibility(flops_per_second)
    report_reduced_exposure_options(flops_per_second)

    _heading("VERDICT")
    if reachable:
        print("At least one model config reaches the 1000-exposure saturating")
        print("regime within a day in this container.")
    else:
        print("NO model config reaches the 1000-exposure saturating regime in a")
        print("reasonable time here. P1's CALIBRATION number requires the Mac mini.")
    print()
    print("* minutes column uses the PESSIMISTIC efficiency estimate.")
    print()
    print("What the container CAN do: validate the sweep machinery end to end at")
    print("reduced exposures, so that when it runs on the Mac mini the only")
    print("unknown left is hardware behaviour. Reduced-exposure numbers must not")
    print("be compared to the 2.0 anchor (mistake M4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
