"""Pattern encoding: deterministic sparse distributed representations (SDRs).

Components 1-2 of the proposal ("Input -> Dynamic Pattern Encoder -> Pattern
Space"): a concept IS a stable sparse activation pattern. This layer is a
deterministic function from input to a sparse binary vector -- no training, no
gradients, no torch.nn.Module -- and is deliberately independent of
src.models (a separate, lower layer).

Public entry point is src.patterns.sdr.
"""

from src.patterns.sdr import (
    SDR,
    N_CANONICAL,
    THETA_MATCH_CANONICAL,
    W_CANONICAL,
    bind,
    bundle,
    capacity_bits,
    capacity_patterns,
    encode_person,
    encode_token,
    expected_overlap,
    false_positive_membership_rate,
    false_positive_probability,
    matches,
    matching_probability_after_flip,
    noisy_copy,
    overlap,
    overlap_set_cardinality,
    sum_overlap_set_cardinalities,
    union,
    union_expected_size,
)

__all__ = [
    "SDR",
    "N_CANONICAL",
    "THETA_MATCH_CANONICAL",
    "W_CANONICAL",
    "bind",
    "bundle",
    "capacity_bits",
    "capacity_patterns",
    "encode_person",
    "encode_token",
    "expected_overlap",
    "false_positive_membership_rate",
    "false_positive_probability",
    "matches",
    "matching_probability_after_flip",
    "noisy_copy",
    "overlap",
    "overlap_set_cardinality",
    "sum_overlap_set_cardinalities",
    "union",
    "union_expected_size",
]
