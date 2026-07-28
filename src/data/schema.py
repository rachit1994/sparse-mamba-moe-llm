"""Shared data contract for the knowledge-capacity experiment.

This file is the interface between the dataset generator, the training loop, and
the bits-recovered metric. It is OWNED BY THE LEAD: agents must not change the
public names, field types, or cardinalities here. If a change is needed, stop and
report -- a silent change breaks the entropy calculation in a way that will not
raise an exception.

Entropy of a dataset of N people is exactly:

    H_total_bits = N * sum(log2(V_j) for V_j in ATTRIBUTE_CARDINALITY.values())

Every downstream number depends on that being right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "ATTRIBUTE_CARDINALITY",
    "ATTRIBUTES",
    "BITS_PER_PERSON",
    "Person",
    "Split",
    "dataset_entropy_bits",
    "bits_per_attribute",
]

# Cardinality of each attribute's value set. These define the dataset entropy and
# therefore the denominator of every capacity claim. Do not edit without updating
# implementation/04_golden_dataset.md and re-deriving every reported number.
ATTRIBUTE_CARDINALITY: Final[dict[str, int]] = {
    "birth_date": 73_000,   # day x year over 1900-2099
    "birth_city": 1_000,
    "university": 300,
    "major": 100,
    "employer": 1_000,
}

ATTRIBUTES: Final[tuple[str, ...]] = tuple(ATTRIBUTE_CARDINALITY)


def bits_per_attribute() -> dict[str, float]:
    """Exact information content of each attribute, in bits."""
    return {k: math.log2(v) for k, v in ATTRIBUTE_CARDINALITY.items()}


BITS_PER_PERSON: Final[float] = sum(bits_per_attribute().values())


def dataset_entropy_bits(n_people: int) -> float:
    """Total entropy of a dataset of `n_people`, in bits.

    Raises:
        ValueError: if n_people is negative.
    """
    if n_people < 0:
        raise ValueError(f"n_people must be non-negative, got {n_people}")
    return n_people * BITS_PER_PERSON


class Split:
    """Dataset splits. See implementation/04_golden_dataset.md section 5."""

    TRAIN: Final[str] = "train"
    PROBE: Final[str] = "probe"                  # same people, held-out paraphrasings
    UNSEEN_PERSON: Final[str] = "unseen_person"  # control: must score ~0
    SEALED: Final[str] = "sealed"                # gate-only, separate seed

    ALL: Final[tuple[str, ...]] = (TRAIN, PROBE, UNSEEN_PERSON, SEALED)


@dataclass(frozen=True, slots=True)
class Person:
    """One synthetic individual.

    Attributes:
        person_id: Unique synthetic name, e.g. "Kelmoran Vushiel-4417". Generated so
            that no real person is representable (contamination proof by construction).
        attributes: Maps every key in ATTRIBUTES to its value. Must be complete --
            a missing key silently changes the entropy and is rejected.
    """

    person_id: str
    attributes: dict[str, str]

    def __post_init__(self) -> None:
        missing = set(ATTRIBUTES) - set(self.attributes)
        if missing:
            raise ValueError(f"Person {self.person_id!r} missing attributes: {sorted(missing)}")
        extra = set(self.attributes) - set(ATTRIBUTES)
        if extra:
            raise ValueError(f"Person {self.person_id!r} has unknown attributes: {sorted(extra)}")
