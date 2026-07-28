"""The knowledge-bits-recovered metric.

This is the headline measurement of the project (see
implementation/00_START_HERE.md section 1 and implementation/04_golden_dataset.md
section 4): how many bits of factual knowledge a model can extract about the
synthetic golden dataset, per parameter it costs to store the model.

    bits_recovered(j) = sum_people [ log2(V_j) - CE_bits(model, correct_value) ]
    bits_per_param    = sum_j bits_recovered(j) / parameter_count

Depends only on ``src/data/schema.py`` (the attribute-cardinality contract). Does
not depend on any dataset generator or model implementation -- callers pass in
already-computed logits/targets and already-instantiated models.

Numerical conventions (implementation/02_testing_philosophy.md section 5; all are
load-bearing, not style choices):

* PyTorch's ``F.cross_entropy`` returns NATS. This module converts to BITS at a
  single named boundary (``_NATS_TO_BITS``) so the conversion cannot be silently
  duplicated, omitted, or applied twice. Omitting it is a 1.4427x inflation --
  the single most likely arithmetic error in this project
  (implementation/03_edge_cases_and_scares.md S1).
* Only answer tokens are scored (``answer_mask``); prompt tokens are excluded
  entirely, not merely down-weighted.
* Reduction is ``sum`` over answer tokens, never ``mean`` -- averaging silently
  divides by sequence length and changes the meaning of the number.
* Cross-entropy is accumulated in float32 regardless of the model's runtime
  dtype (bf16/fp16 lack the precision to accumulate over many tokens, and the
  target hardware's MPS backend does not support float64, so float32 is the
  floor *and* the ceiling here -- never promote to float64).
* Per-item bit contributions are clamped at 0 (a model can be worse than
  chance; negative "stored knowledge" is meaningless and would silently offset
  real gains elsewhere).

Two assertions are live in code, not only in tests (implementation/04_golden_dataset.md
section 4):

1. ``bits_recovered <= dataset entropy`` -- enforced in :func:`bits_recovered_per_attribute`
   at the tightest possible granularity (per person, per attribute): a
   cross-entropy can never be negative, so any negative or non-finite per-item
   CE_bits value would imply recovering more bits from one draw than that draw
   can contain. This is the highest-value assertion in the project -- if it
   fires, the metric (or its input) is broken, not the model.
2. ``bits_recovered >= 0`` per attribute -- guaranteed by the clamp above and
   re-asserted defensively after aggregation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import ATTRIBUTE_CARDINALITY

__all__ = [
    "EntropyBoundViolation",
    "answer_cross_entropy_bits",
    "bits_recovered_per_attribute",
    "bits_per_parameter",
    "count_parameters",
]

# The single named nats->bits conversion boundary. Every CE_bits value in this
# module passes through this constant exactly once. Never inline `/math.log(2)`
# or `* 1.4427` anywhere else in this file or its callers.
_NATS_TO_BITS: float = 1.0 / math.log(2.0)

# Absolute tolerance on the entropy-bound check, in bits. Must comfortably
# exceed float32 rounding noise in log_softmax (~1e-6 relative) while staying
# many orders of magnitude below any real bug (the nats/bits error alone is a
# 1.4427x multiplicative error -- see S1 in implementation/03_edge_cases_and_scares.md).
_ENTROPY_BOUND_EPS_BITS: float = 1e-4


class EntropyBoundViolation(ValueError):
    """Raised when a measurement implies recovering more bits than exist.

    This is the runtime form of the project's highest-value assertion
    (implementation/04_golden_dataset.md section 4, implementation/02_testing_philosophy.md
    section 3): cross-entropy is mathematically non-negative, so a negative or
    non-finite per-item CE_bits value would mean that single (person, attribute)
    draw recovered more than log2(V) bits -- more than that one draw can
    contain. A violation means the metric or its input is broken; it is never a
    legitimate model result and must never be silently clamped away.
    """


def answer_cross_entropy_bits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    answer_mask: torch.Tensor,
) -> float:
    """Summed cross-entropy over answer tokens only, in bits.

    Prompt tokens (``answer_mask == False``) are excluded entirely: their
    values in ``logits``/``targets`` may be anything, including garbage or
    non-finite placeholders, without affecting the result. Answer tokens
    (``answer_mask == True``) must be well-formed: finite logits and in-range
    target indices.

    ``logits`` and ``targets`` must already be aligned so that
    ``logits[..., t, :]`` is the model's prediction for ``targets[..., t]``
    (i.e. any causal-LM shift-by-one must already be applied by the caller;
    this function does not shift).

    Args:
        logits: unnormalised scores, shape ``(..., seq_len, vocab_size)``, any
            floating dtype (fp32/fp16/bf16). Computation is done in float32
            regardless of input dtype.
        targets: target token ids, shape ``(..., seq_len)``, integer dtype.
        answer_mask: boolean (or 0/1-castable) mask, shape ``(..., seq_len)``.
            Non-zero/True marks a token to score.

    Returns:
        Total cross-entropy of the answer tokens, summed (not averaged), in
        bits. Always finite and non-negative.

    Raises:
        ValueError: if ``logits`` has fewer than 2 dimensions, if the shapes of
            ``logits``/``targets``/``answer_mask`` are mutually incompatible
            (message includes the actual shapes), if ``answer_mask`` selects
            zero tokens, if target indices are out of ``[0, vocab_size)`` for a
            scored token, or if a scored token's logits contain a non-finite
            (inf/nan) value.
        TypeError: if ``targets`` is a floating-point tensor (target token ids
            must be integral; a float tensor here is almost always a caller
            bug, e.g. passing probabilities instead of ids).
    """
    if logits.dim() < 2:
        raise ValueError(
            f"logits must have at least 2 dims (..., vocab_size), got shape {tuple(logits.shape)}"
        )
    if targets.shape != answer_mask.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} != answer_mask shape {tuple(answer_mask.shape)}"
        )
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} incompatible with targets shape "
            f"{tuple(targets.shape)}: expected logits.shape[:-1] == targets.shape"
        )
    if torch.is_floating_point(targets):
        raise TypeError(
            f"targets must be an integer dtype (token ids), got floating dtype {targets.dtype}"
        )

    mask_flat = answer_mask.reshape(-1).bool()
    if not bool(mask_flat.any()):
        # A silent 0.0 here would read as "the model knows nothing", which is a
        # different claim from "nothing was scored". Refuse to be ambiguous
        # (implementation/03_edge_cases_and_scares.md, Metric / Empty).
        raise ValueError("answer_mask selects zero tokens; nothing to score")

    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)[mask_flat].float()
    flat_targets = targets.reshape(-1)[mask_flat].long()

    tgt_min = int(flat_targets.min())
    tgt_max = int(flat_targets.max())
    if tgt_min < 0 or tgt_max >= vocab_size:
        raise ValueError(
            f"targets out of range [0, {vocab_size}) on scored (answer) tokens: "
            f"min={tgt_min}, max={tgt_max}"
        )
    if not bool(torch.isfinite(flat_logits).all()):
        raise ValueError(
            "logits contain non-finite (inf/nan) values on a scored (answer) token; "
            "refusing to score. Non-finite values on masked-out (prompt) tokens are fine."
        )

    # reduction="sum", never "mean": mean silently divides by the number of
    # answer tokens, which changes the meaning of the number
    # (implementation/02_testing_philosophy.md section 5). Computed and
    # accumulated in float32 (flat_logits was cast with .float() above).
    total_nats = F.cross_entropy(flat_logits, flat_targets, reduction="sum")

    # The nats->bits conversion. This is the ONLY place it happens in this
    # module (implementation/03_edge_cases_and_scares.md S1).
    total_bits = float(total_nats) * _NATS_TO_BITS
    return total_bits


def bits_recovered_per_attribute(model_ce_bits: dict[str, list[float]]) -> dict[str, float]:
    """Aggregate per-item cross-entropy into bits recovered, per attribute.

    For attribute ``j`` with cardinality ``V_j`` (from
    ``src.data.schema.ATTRIBUTE_CARDINALITY``) and one CE_bits value per scored
    person:

        bits_recovered(j) = sum_people clamp(log2(V_j) - CE_bits(person), min=0)

    Clamping happens per person, before summing, per
    implementation/04_golden_dataset.md section 4: a model can score worse than
    chance on an individual item, but a negative per-item contribution would
    silently offset genuine recovered knowledge elsewhere in the same
    attribute if left unclamped.

    Note on duplicates: this function receives already-reduced scalar CE_bits
    values with no person identifier attached, so it cannot detect whether the
    same person was scored twice for an attribute (implementation/03_edge_cases_and_scares.md,
    Metric / Duplicate). Callers are responsible for ensuring each list entry
    corresponds to exactly one (person, attribute) pair; double-counting a
    person inflates the result linearly and this function has no way to catch
    it from the data it is given.

    Args:
        model_ce_bits: maps attribute name (must be a key of
            ``src.data.schema.ATTRIBUTE_CARDINALITY``) to a list of per-person
            CE_bits values (as returned by :func:`answer_cross_entropy_bits`).

    Returns:
        Maps each input attribute name to its total bits recovered. Every
        value is finite and >= 0.

    Raises:
        ValueError: if an attribute name is not in
            ``src.data.schema.ATTRIBUTE_CARDINALITY``, or if an attribute maps
            to an empty list (a silent 0.0 would read as "the model knows
            nothing about this attribute" rather than "zero people were
            scored").
        EntropyBoundViolation: if any per-item CE_bits value is negative or
            non-finite. Cross-entropy is mathematically non-negative; a
            negative value implies that single item alone recovered more bits
            than log2(V_j) -- more than exist for that one draw. This is the
            project's headline "bits_recovered <= dataset entropy" assertion,
            enforced at the finest granularity that makes it actually
            reachable and testable.
    """
    result: dict[str, float] = {}
    for attribute, ce_values in model_ce_bits.items():
        if attribute not in ATTRIBUTE_CARDINALITY:
            raise ValueError(
                f"unknown attribute {attribute!r}; expected one of "
                f"{sorted(ATTRIBUTE_CARDINALITY)}"
            )
        if len(ce_values) == 0:
            raise ValueError(
                f"model_ce_bits[{attribute!r}] is empty; cannot compute bits recovered "
                f"for zero scored people"
            )

        ceiling_per_item = math.log2(ATTRIBUTE_CARDINALITY[attribute])
        ce_arr = np.asarray(ce_values, dtype=np.float32)

        bad = ~np.isfinite(ce_arr) | (ce_arr < 0.0)
        if np.any(bad):
            bad_indices = np.nonzero(bad)[0]
            first = int(bad_indices[0])
            raise EntropyBoundViolation(
                f"attribute {attribute!r}: item {first} has CE_bits={ce_arr[first]!r}, which "
                f"is negative or non-finite. Cross-entropy cannot be negative; this would mean "
                f"recovering more than log2({ATTRIBUTE_CARDINALITY[attribute]})="
                f"{ceiling_per_item:.4f} bits from a single item, violating "
                f"bits_recovered <= dataset entropy ({len(bad_indices)} of {len(ce_values)} "
                f"items affected)."
            )

        per_item = np.clip(ceiling_per_item - ce_arr, a_min=0.0, a_max=None)
        attribute_total = float(per_item.sum(dtype=np.float32))

        # Defence in depth: given the per-item check above, this is
        # mathematically implied and should be unreachable. Kept as a second
        # layer against a logic bug in this function itself, not as the
        # primary enforcement path (which is the per-item check above -- see
        # EntropyBoundViolation's docstring for why post-hoc aggregate checks
        # alone are not sensitive enough).
        ceiling_total = ceiling_per_item * len(ce_values)
        if attribute_total > ceiling_total + _ENTROPY_BOUND_EPS_BITS:
            raise EntropyBoundViolation(
                f"attribute {attribute!r}: bits_recovered={attribute_total} exceeds dataset "
                f"entropy ceiling {ceiling_total} for {len(ce_values)} scored people "
                f"(internal invariant violation)."
            )
        assert attribute_total >= 0.0, (
            f"attribute {attribute!r}: clamped per-item contributions summed negative "
            f"({attribute_total}); this cannot happen given the clip above and indicates a "
            f"logic bug in this function."
        )

        result[attribute] = attribute_total
    return result


def bits_per_parameter(total_bits_recovered: float, parameter_count: int) -> float:
    """Normalise total bits recovered by parameter count.

    Args:
        total_bits_recovered: sum of :func:`bits_recovered_per_attribute` values
            across all attributes. Must be non-negative (every attribute total
            is clamped at 0 before this point).
        parameter_count: as returned by :func:`count_parameters`. Must be
            positive.

    Returns:
        Bits recovered per parameter.

    Raises:
        ValueError: if ``parameter_count`` is not positive, or if
            ``total_bits_recovered`` is negative (which cannot happen if it was
            built from :func:`bits_recovered_per_attribute` outputs, and
            indicates a caller bug if it does).
    """
    if parameter_count <= 0:
        raise ValueError(f"parameter_count must be positive, got {parameter_count}")
    if total_bits_recovered < 0:
        raise ValueError(
            f"total_bits_recovered must be non-negative (every attribute's contribution is "
            f"clamped at 0 by bits_recovered_per_attribute before summing); got "
            f"{total_bits_recovered}"
        )
    return float(total_bits_recovered) / float(parameter_count)


def count_parameters(model: torch.nn.Module, include_embeddings: bool = True) -> int:
    """Count parameters in ``model`` under one fixed, stated convention.

    Convention (must be identical across every architecture compared, and
    identical to the external 2.0 bits/param baseline it is compared against
    -- implementation/02_testing_philosophy.md section 5, implementation/03_edge_cases_and_scares.md
    S3):

    * Only parameters with ``requires_grad=True`` are counted ("trainable"),
      matching Allen-Zhu & Li's stated convention.
    * Tied parameters (e.g. an embedding weight-tied to the output projection)
      are counted once. This falls out of
      ``nn.Module.named_parameters(remove_duplicate=True)``, which is torch's
      default and de-duplicates by parameter object identity, not by name.
    * Embedding parameters are identified structurally -- by iterating
      ``model.modules()`` and matching ``isinstance(module, (nn.Embedding,
      nn.EmbeddingBag))`` -- never by parameter name substring matching, which
      is fragile across architectures (a Mamba/MoE model may name its
      embedding table anything).

    Args:
        model: the model to count. Must be a ``torch.nn.Module``.
        include_embeddings: if ``False``, subtract parameters belonging to any
            ``nn.Embedding``/``nn.EmbeddingBag`` module from the total. If the
            same parameter is used both as an embedding table and (via weight
            tying) as the output projection, it is still excluded entirely --
            it is structurally an embedding parameter.

    Returns:
        Total parameter count, >= 0.

    Raises:
        TypeError: if ``model`` is not a ``torch.nn.Module``.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")

    embedding_param_ids: set[int] = set()
    if not include_embeddings:
        for module in model.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for p in module.parameters(recurse=False):
                    embedding_param_ids.add(id(p))

    total = 0
    for _, param in model.named_parameters(remove_duplicate=True):
        if not param.requires_grad:
            continue
        if id(param) in embedding_param_ids:
            continue
        total += param.numel()
    return total
