"""Model configuration and analytic parameter counting for the baseline transformer.

This file is the interface between the architecture (`src/models/tiny_lm.py`)
and every place that needs to know a model's parameter count *before* paying
the cost of instantiating it -- most importantly the capacity sweep, which
must pick dataset sizes N from the target parameter count (see
implementation/04_golden_dataset.md section 2) without building every
candidate model.

Parameter-counting convention (implementation/03_edge_cases_and_scares.md
S3, implementation/02_testing_philosophy.md section 5) -- **stated explicitly
because a mismatched convention silently manufactures a 2x difference**:

* All *trainable* parameters are counted (nothing is frozen in this
  architecture, so this is simply "every parameter").
* A weight-tied embedding/output-projection pair is counted **once**, not
  twice: :func:`analytic_param_count` below never adds the output
  projection's own vocab_size*d_model term when ``tie_embeddings=True``.
* This must equal -- and is asserted equal to, in
  ``tests/test_models_tiny_lm.py`` -- the actual count returned by
  ``src.metrics.bits.count_parameters(model, include_embeddings=True)``,
  which independently arrives at the same number by iterating the
  instantiated model's ``named_parameters(remove_duplicate=True)``. Two
  independently-derived numbers agreeing is the actual evidence; either one
  alone is not (implementation/02_testing_philosophy.md section 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "ModelConfig",
    "PRESET_CONFIGS",
    "analytic_param_count",
]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hyperparameters of the baseline decoder-only transformer (`TinyLM`).

    Fixed architectural choices that are *not* config fields (kept out of
    this dataclass deliberately -- this is the "boring control arm", not a
    general-purpose architecture search space, per B3_baseline_model.md /
    section 4 of AGENTS.md: no speculative configurability):

    * MLP hidden width is always ``4 * d_model`` (standard transformer ratio).
    * Attention is standard causal self-attention (no rotary/ALiBi/relative
      position variants); position is a learned absolute embedding table.
    * All linear layers except the (untied) output head carry a bias.
    * No dropout (this project's overfit test wants a deterministic,
      un-regularised memoriser; dropout would also reintroduce randomness
      into the forward pass that the resume/determinism tests would then
      have to account for, for no benefit to a control-arm baseline).

    Attributes:
        n_layer: number of transformer blocks. Must be a positive int.
        n_head: number of attention heads. Must be a positive int and must
            evenly divide ``d_model`` (each head gets ``d_model // n_head``
            dimensions).
        d_model: residual-stream / embedding width. Must be a positive int.
        vocab_size: size of the token vocabulary. Must be a positive int.
        block_size: maximum context length (the positional embedding table
            has exactly this many rows). Must be an int >= 2 (a causal
            shift-by-one, used throughout this project to build
            next-token targets, needs at least 2 positions to produce one
            target).
        tie_embeddings: if True, the output projection (`lm_head`) reuses
            the token-embedding matrix instead of learning its own
            ``vocab_size x d_model`` weight (standard weight tying).
    """

    n_layer: int
    n_head: int
    d_model: int
    vocab_size: int
    block_size: int
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for field_name in ("n_layer", "n_head", "d_model", "vocab_size", "block_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
        if isinstance(self.tie_embeddings, int) and not isinstance(self.tie_embeddings, bool):
            raise TypeError(f"tie_embeddings must be a bool, got {type(self.tie_embeddings).__name__}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size < 2:
            raise ValueError(
                f"block_size must be >= 2 (a shift-by-one target needs at least 2 "
                f"positions), got {self.block_size}"
            )
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by n_head ({self.n_head})"
            )


def analytic_param_count(config: ModelConfig) -> int:
    """Closed-form parameter count for a `TinyLM(config)`, without instantiating it.

    Requirement R1 (implementation/06_agent_briefs/B3_baseline_model.md):
    parameter count is the denominator of the headline bits/param metric, so
    it must be computable from the config alone and then asserted equal to
    the instantiated model's actual count -- never trusted from one source
    alone (implementation/03_edge_cases_and_scares.md S3).

    Derivation, with ``d = config.d_model``:

    * token embedding (`wte`): ``vocab_size * d``
    * positional embedding (`wpe`): ``block_size * d``
    * final layernorm (`ln_f`): ``2 * d`` (weight + bias)
    * per transformer block (`12*d^2 + 13*d`):
        - ``ln_1``: ``2*d``
        - attention QKV projection (``Linear(d, 3d)``, with bias): ``3*d^2 + 3*d``
        - attention output projection (``Linear(d, d)``, with bias): ``d^2 + d``
        - ``ln_2``: ``2*d``
        - MLP up-projection (``Linear(d, 4d)``, with bias): ``4*d^2 + 4*d``
        - MLP down-projection (``Linear(4d, d)``, with bias): ``4*d^2 + d``
        - sum: ``(2)+(3d^2+3)+(d^2+1)+(2)+(4d^2+4)+(4d^2+1)`` in units of d
          and d^2 -> ``12*d^2 + 13*d``
    * output head (`lm_head`, no bias): ``0`` if ``tie_embeddings`` (it reuses
      `wte`'s weight, already counted), else ``vocab_size * d``.

    Args:
        config: the model configuration to count.

    Returns:
        Total parameter count under the "all trainable, tied params counted
        once" convention (module docstring above).
    """
    d = config.d_model
    per_layer = 12 * d * d + 13 * d
    total = d * (config.vocab_size + config.block_size + 2) + config.n_layer * per_layer
    if not config.tie_embeddings:
        total += config.vocab_size * d
    return total


# Reference configs spanning the capacity sweep's three anchor points
# (implementation/04_golden_dataset.md section 2: ~1M / ~10M / ~100M params).
# vocab_size=256 throughout: a byte-level vocabulary (see
# src/train.py:make_toy_dataset and the module docstring there for why no
# tokenizer dependency is introduced at this layer) tokenises every possible
# input -- including every synthetic name from src/data/generate.py --
# byte-identically and with no OOV handling required.
#
# "tiny" is not part of the ~1M/10M/100M sweep; it exists purely so the
# required tests (determinism, resume, overfit, param-count) run in seconds
# on a 4-core CPU container with no GPU (R8).
PRESET_CONFIGS: Final[dict[str, ModelConfig]] = {
    "tiny": ModelConfig(n_layer=2, n_head=2, d_model=32, vocab_size=32, block_size=16, tie_embeddings=True),
    "1m": ModelConfig(n_layer=5, n_head=4, d_model=128, vocab_size=256, block_size=256, tie_embeddings=True),
    "10m": ModelConfig(n_layer=6, n_head=6, d_model=384, vocab_size=256, block_size=256, tie_embeddings=True),
    "100m": ModelConfig(n_layer=8, n_head=16, d_model=1024, vocab_size=256, block_size=512, tie_embeddings=True),
}
