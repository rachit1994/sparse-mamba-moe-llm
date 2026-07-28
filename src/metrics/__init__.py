"""Metrics for the knowledge-capacity experiment.

Public entry point is the bits-recovered metric in :mod:`src.metrics.bits`.
"""

from src.metrics.bits import (
    EntropyBoundViolation,
    answer_cross_entropy_bits,
    bits_per_parameter,
    bits_recovered_per_attribute,
    count_parameters,
)

__all__ = [
    "EntropyBoundViolation",
    "answer_cross_entropy_bits",
    "bits_per_parameter",
    "bits_recovered_per_attribute",
    "count_parameters",
]
