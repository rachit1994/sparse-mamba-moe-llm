"""Synthetic biography dataset for the knowledge-capacity experiment.

Public entry point is the generator in :mod:`src.data.generate`. The data
contract (attribute cardinalities, ``Person``, splits) lives in
:mod:`src.data.schema` and is lead-owned.
"""

from src.data.generate import (
    generate_people,
    render_probe_qa,
    render_training_text,
    write_dataset,
)

__all__ = [
    "generate_people",
    "render_probe_qa",
    "render_training_text",
    "write_dataset",
]
