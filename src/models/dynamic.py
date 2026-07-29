"""DynamicSubstrate: the thesis arm. Knowledge in state, not in weights.

THE QUESTION this file answers: can a FIXED substrate acquire and retrieve
facts via a write path that never takes a gradient step on its base weights?
implementation/08_thesis_alignment.md section 6 states the requirement this
class exists to satisfy: "the dynamic arm must be able to acquire knowledge
without a gradient step on base weights -- via test-time state updates
(TTT/Titans-style) or Hebbian synaptic change (BDH-style). If it can only
learn by SGD into weights, it is a transformer with extra steps."

HOW TO REVIEW IT:
  1. `base_weight_tensors()` returns exactly the arrays that must be
     bit-identical before and after any sequence of `write()` calls. Confirm
     `write()` (below) never assigns to `self._context_projection` or
     `self._offset_projection` -- it only ever mutates `self._association_matrix`.
  2. Both are plain numpy arrays, never `torch.nn.Parameter`s and never
     touched inside a `torch.autograd` context anywhere in this file -- there
     is no gradient *concept* applicable to them, not merely an untaken one.
  3. `substrate_bytes_breakdown()` is built directly from
     `base_weight_tensors()` and `persistent_state_tensors()`: there is no
     third place a tensor could be silently excluded from the count.

MECHANISM (hetero-associative Hebbian memory; the brief's suggested
Hopfield/fast-weight design, adopted as-is since it satisfies the gate
structurally per point 2 above):

  * `_context_projection` (vocab_size x code_dim) and `_offset_projection`
    (max_answer_offset x code_dim) are FIXED random {-1,+1} matrices drawn
    once at construction from `seed` -- the "fixed substrate" of the thesis.
  * A fact is a (token_ids, answer_mask) pair, exactly
    `src.data.tokenizer.Tokenizer.encode_qa`'s return shape: the question is
    every token with `answer_mask=False`, the answer every token with
    `answer_mask=True`, in order.
  * The KEY for the o-th answer token BINDS a context code and an offset
    code via elementwise product (Kanerva-style bipolar binding, not
    addition): `context_code = sign(sum of context tokens' rows of
    _context_projection)`, then `key = context_code * _offset_projection[o]`.
    Binding by elementwise product, not by summing the two and re-quantising,
    is deliberate and was verified necessary, not stylistic: a context with
    many tokens produces a sum whose magnitude (up to len(context)) swamps a
    single +-1 offset row, so summing-then-sign collapses different offsets'
    keys onto nearly the same vector for a shared context (measured: 89%
    bit-agreement between offset 0 and offset 1 for a 9-token context) --
    every answer position then reads back the same, most-frequently-written
    row. Elementwise product gives the offset exactly as much influence as
    the context in every dimension regardless of context length. Bipolar
    keys, not one-hot or dense reals, are the deliberate choice: it is what
    makes the Amit-Gutfreund-Sompolinsky ~0.138*N capacity anchor apply to
    `code_dim` at all -- a dense real-valued key has no such anchor.
  * WRITE is pure Hebbian accumulation, no gradient, no optimiser:
    `_association_matrix += outer(bipolar_value(token_id), key)`.
  * READ is one linear pass, no attractor iteration: `_association_matrix @ key`
    -- a deliberate scope cut, not an oversight (see the class docstring's
    Non-goals).

Depends on nothing outside `numpy`/`torch`; does not import
`src.data.tokenizer` or `src.data.generate` -- a fact is already token ids by
the time it reaches this module, matching how `TinyLM.forward` never imports
the tokenizer either.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["DynamicSubstrate", "Fact"]

# One (token_ids, answer_mask) pair -- exactly Tokenizer.encode_qa's return
# shape (src/data/tokenizer.py, "USE, DO NOT MODIFY"). answer_mask[i]==True
# marks ids[i] as an answer token to be written/read; the rest is context.
Fact = tuple[Sequence[int], Sequence[bool]]

# Upper bound on answer length in tokens. Provenance: the golden dataset's
# longest answer is birth_date ("<day> <Month> <year>"), which the shared
# word-level tokenizer (src/data/tokenizer.py) splits into up to 2 day-digits
# + 1 month word + 4 year-digits = 7 tokens (verified against
# src/data/tokenizer.py's digit-splitting rule). 16 leaves >2x headroom for
# any answer this dataset can produce without silently truncating one.
_MAX_ANSWER_OFFSET: int = 16

# Bytes per element of the fixed random projections: bipolar {-1,+1} values
# fit in a single signed byte, and there is nothing to gain from a wider
# dtype for a two-valued quantity.
_PROJECTION_ITEMSIZE_BYTES: int = 1

# Bytes per element of the Hebbian accumulator. float32 (never float64,
# matching src/metrics/bits.py's stated float32 floor-and-ceiling convention)
# -- accumulated outer products of +-1 values stay small and well within
# float32's range for any fact count this project runs.
_ASSOCIATION_ITEMSIZE_BYTES: int = 4

# Sentinel used ONLY in read_logits' padded batch input to mark a position
# past a row's true length. Deliberately not 0 or any real token id: numpy
# fancy-indexing silently wraps a genuine -1 to "last row" instead of
# raising, which is exactly the untrusted-input footgun this module must not
# have (edge-case ladder rung 9) -- so real token ids are required to be >= 0
# and PAD_ID is the one reserved negative value, checked explicitly rather
# than relied upon to fail loudly on its own.
PAD_ID: int = -1


def _bipolar_projection(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """A fixed random {-1,+1} matrix -- one "layer" of the substrate `write()` never touches."""
    return (rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1).astype(np.int8)


def _split_fact(ids: Sequence[int], answer_mask: Sequence[bool], fact_index: int) -> tuple[list[int], list[int]]:
    """Split one fact into (context_ids, answer_ids), validating shape and non-emptiness.

    Raises:
        ValueError: if `ids`/`answer_mask` differ in length, or no position is
            marked as an answer (a silent empty answer would make `write` a
            silent no-op and `read_logits` unable to say what was asked --
            implementation/03_edge_cases_and_scares.md, Metric / Empty).
    """
    if len(ids) != len(answer_mask):
        raise ValueError(
            f"fact {fact_index}: ids has length {len(ids)} but answer_mask has "
            f"length {len(answer_mask)}"
        )
    context_ids = [int(i) for i, m in zip(ids, answer_mask) if not m]
    answer_ids = [int(i) for i, m in zip(ids, answer_mask) if m]
    if not answer_ids:
        raise ValueError(f"fact {fact_index}: answer_mask selects zero tokens; nothing to write")
    return context_ids, answer_ids


class DynamicSubstrate:
    """A fixed-size associative memory whose knowledge lives in accumulated state.

    Non-goals (deliberate, stated so a reviewer does not read them as gaps):
      * No attractor iteration on read (the brief's "optionally iterate to
        convergence"). A single linear pass is enough to demonstrate the
        gate (retrieval above chance without touching base weights); adding
        iteration would grow this file without changing what the gate proves,
        and it is easy to add later behind the same `read_logits` signature.
      * No concurrency contract. N/A: this is an in-process numpy object with
        no thread-safety claim, matching the container's single-threaded
        harness -- same posture as `TinyLM`, which makes no such claim either.
      * Bag-of-words context keys carry no word order. This is a real,
        intentional source of interference (two facts whose questions share
        the same token multiset collide) and is exactly what the capacity
        test in tests/test_models_dynamic.py is measuring, not a bug to hide.

    Complexity: `write` is O(F * L * D) time where F = facts, L = mean answer
    length in tokens, D = code_dim (one O(D) key + O(V*D) outer-product update
    per answer token, V = vocab_size dominates the constant but not the scaling
    in F or L). `read_logits` is O(B * S * (D + V*D)) for a (B, S) batch. Space
    is O(V*D) for the association matrix, dominant over the O(V*D + K*D)
    fixed projections (K = _MAX_ANSWER_OFFSET).
    """

    def __init__(self, substrate_bytes_budget: int, seed: int, vocab_size: int) -> None:
        """Allocate a fixed substrate sized to fit `substrate_bytes_budget`.

        `vocab_size` is not in the brief's literal 2-argument signature; it is
        added because `read_logits` must produce a score for every vocabulary
        entry and there is no other channel by which this class could learn
        the vocabulary's size (`TinyLM` gets the equivalent value from
        `ModelConfig.vocab_size`, its own construction-time argument -- this
        is the same pattern, not a new one). Stated once here per the
        project's divergence-naming rule; the two originally-specified
        arguments keep their names, types, and order.

        Args:
            substrate_bytes_budget: total bytes available for every persistent
                tensor this object owns (see `substrate_bytes`). Determines
                `code_dim`, the memory's pattern dimension, by solving
                `code_dim * bytes_per_code_dim <= substrate_bytes_budget` for
                the largest integer `code_dim`. Must be large enough for
                `code_dim >= 1`.
            seed: determinism seed for the fixed random projections. Same
                (seed, vocab_size, budget) always yields bit-identical
                `base_weight_tensors()`.
            vocab_size: size of the token vocabulary `read_logits` scores
                over. Must match the tokenizer that produced every fact's ids.

        Raises:
            TypeError: if any argument is not a plain int (bools rejected, as
                elsewhere in this project's constructors).
            ValueError: if `seed < 0`, `vocab_size <= 0`, or
                `substrate_bytes_budget` is too small to afford `code_dim=1`.
        """
        for name, value in (
            ("substrate_bytes_budget", substrate_bytes_budget),
            ("seed", seed),
            ("vocab_size", vocab_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        bytes_per_code_dim = (
            vocab_size * _PROJECTION_ITEMSIZE_BYTES  # context projection column
            + _MAX_ANSWER_OFFSET * _PROJECTION_ITEMSIZE_BYTES  # offset projection column
            + vocab_size * _ASSOCIATION_ITEMSIZE_BYTES  # association matrix column
        )
        code_dim = substrate_bytes_budget // bytes_per_code_dim
        if code_dim < 1:
            raise ValueError(
                f"substrate_bytes_budget={substrate_bytes_budget} is too small for "
                f"vocab_size={vocab_size}: needs at least {bytes_per_code_dim} bytes "
                f"for code_dim=1 (got room for code_dim={code_dim})"
            )

        self._vocab_size = vocab_size
        self._code_dim = code_dim
        rng = np.random.default_rng(seed)
        self._context_projection = _bipolar_projection(rng, (vocab_size, code_dim))
        self._offset_projection = _bipolar_projection(rng, (_MAX_ANSWER_OFFSET, code_dim))
        self._association_matrix = np.zeros((vocab_size, code_dim), dtype=np.float32)

    @property
    def code_dim(self) -> int:
        """The memory's pattern dimension, derived from the byte budget at construction."""
        return self._code_dim

    def base_weight_tensors(self) -> dict[str, np.ndarray]:
        """Tensors `write()` must never change (the gate's hash target)."""
        return {
            "context_projection": self._context_projection,
            "offset_projection": self._offset_projection,
        }

    def persistent_state_tensors(self) -> dict[str, np.ndarray]:
        """Tensors that DO change under `write()` and must be counted in `substrate_bytes`."""
        return {"association_matrix": self._association_matrix}

    def substrate_bytes_breakdown(self) -> dict[str, int]:
        """Byte size of every persistent tensor this object owns, named for a reviewer."""
        tensors = {**self.base_weight_tensors(), **self.persistent_state_tensors()}
        return {name: array.nbytes for name, array in tensors.items()}

    def substrate_bytes(self) -> int:
        """Total bytes: weights + persistent state + index, per the brief's accounting rule."""
        return sum(self.substrate_bytes_breakdown().values())

    def _encode_key(self, context_ids: Sequence[int], offset: int) -> np.ndarray:
        """Bipolar key for the `offset`-th answer token of a question over `context_ids`.

        `key = context_code * offset_projection[offset]`, an elementwise
        (Kanerva-style) binding of two bipolar codes -- see the module
        docstring's MECHANISM section for why this must be a product, not a
        sum-then-requantise: the latter lets a long context's magnitude drown
        out the offset row, collapsing distinct answer positions onto the
        same key.

        Raises:
            ValueError: if `offset` is outside `[0, _MAX_ANSWER_OFFSET)`, or any
                id in `context_ids` is outside `[0, vocab_size)`.
        """
        if not 0 <= offset < _MAX_ANSWER_OFFSET:
            raise ValueError(
                f"answer offset {offset} outside [0, {_MAX_ANSWER_OFFSET}); this "
                f"substrate cannot address an answer token this far into the answer"
            )
        if context_ids:
            bad = [i for i in context_ids if not 0 <= i < self._vocab_size]
            if bad:
                raise ValueError(f"context token id(s) {bad[:5]} outside [0, {self._vocab_size})")
            context_raw = self._context_projection[context_ids].astype(np.int32).sum(axis=0)
            context_code = np.where(context_raw >= 0, 1, -1).astype(np.int8)
        else:
            context_code = np.ones(self._code_dim, dtype=np.int8)
        return context_code * self._offset_projection[offset]

    def write(self, facts: Sequence[Fact]) -> None:
        """Hebbian-accumulate `facts` into persistent state. NO gradient on base weights.

        For every fact and every answer token at offset `o`, accumulates
        `outer(bipolar_one_hot(token_id), key(context, o))` into
        `_association_matrix`. Only `_association_matrix` (returned by
        `persistent_state_tensors`) is ever assigned to; `_context_projection`
        and `_offset_projection` (returned by `base_weight_tensors`) are read,
        never written, in this method -- this is what the gate test asserts.

        Args:
            facts: zero or more `(ids, answer_mask)` pairs. Empty is a valid
                no-op (edge-case ladder rung 1).

        Raises:
            ValueError: if any fact's `ids`/`answer_mask` lengths differ, has
                zero answer tokens, has more answer tokens than
                `_MAX_ANSWER_OFFSET`, or contains a token id outside
                `[0, vocab_size)`.
        """
        for fact_index, (ids, answer_mask) in enumerate(facts):
            context_ids, answer_ids = _split_fact(ids, answer_mask, fact_index)
            if len(answer_ids) > _MAX_ANSWER_OFFSET:
                raise ValueError(
                    f"fact {fact_index}: answer has {len(answer_ids)} tokens, exceeding "
                    f"this substrate's max_answer_offset={_MAX_ANSWER_OFFSET}"
                )
            for offset, token_id in enumerate(answer_ids):
                if not 0 <= token_id < self._vocab_size:
                    raise ValueError(
                        f"fact {fact_index}: answer token id {token_id} outside "
                        f"[0, {self._vocab_size})"
                    )
                key = self._encode_key(context_ids, offset)
                value = -np.ones(self._vocab_size, dtype=np.float32)
                value[token_id] = 1.0
                self._association_matrix += np.outer(value, key)

    def read_logits(self, ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        """Score every position of a padded batch, matching TinyLM's hook shapes.

        Unlike `TinyLM.forward`, this is not next-token prediction: position
        `t` where `answer_mask[..., t]` is True is scored as the substrate's
        recall of that position's OWN token given the question (positions
        with `answer_mask=False`), not the token that follows it. No shift is
        needed before calling `src.metrics.bits.answer_cross_entropy_bits`
        (pass `ids` itself as `targets`) -- see the module docstring's
        Reason section 5 of implementation/08_thesis_alignment.md: forcing
        next-token prediction onto the dynamic arm would assume the
        conclusion the control arm exists to test.

        Padding: rows shorter than the batch's max length must be right-padded
        with `PAD_ID` (-1) in `ids`, with `answer_mask=False` at every padded
        position. Padding must come only after a row's real content, never
        interleaved with it -- otherwise a pad position could be counted as
        context and corrupt that row's keys.

        Args:
            ids: token ids, shape `(batch, seq_len)`, integer dtype. Real
                tokens are `>= 0`; padded positions are `PAD_ID`.
            answer_mask: boolean mask, shape `(batch, seq_len)`. True marks a
                position to be scored as an answer token.

        Returns:
            Logits, shape `(batch, seq_len, vocab_size)`, dtype `torch.float32`.
            Values at `answer_mask=False` positions are `0.0` and must not be
            scored (matches `answer_cross_entropy_bits`'s contract: unmasked
            positions may hold any value).

        Raises:
            TypeError: if `ids` is a floating-point or bool-dtype tensor.
            ValueError: if `ids`/`answer_mask` are not 2D with matching
                shapes, if `batch == 0` or `seq_len == 0`, if a position
                marked True in `answer_mask` holds `PAD_ID`, or if an
                answer's running offset within its row would exceed
                `_MAX_ANSWER_OFFSET` or its token id is outside
                `[0, vocab_size)`.
        """
        if torch.is_floating_point(ids) or ids.dtype == torch.bool:
            raise TypeError(f"ids must be an integer-dtype tensor of token ids, got {ids.dtype}")
        if ids.shape != answer_mask.shape:
            raise ValueError(f"ids shape {tuple(ids.shape)} != answer_mask shape {tuple(answer_mask.shape)}")
        if ids.dim() != 2:
            raise ValueError(f"ids must be 2D (batch, seq_len), got shape {tuple(ids.shape)}")
        batch, seq_len = ids.shape
        if batch == 0:
            raise ValueError("ids has batch size 0; nothing to run the substrate on")
        if seq_len == 0:
            raise ValueError("ids has seq_len 0; nothing to run the substrate on")

        ids_np = ids.cpu().numpy()
        mask_np = answer_mask.cpu().numpy().astype(bool)
        logits = np.zeros((batch, seq_len, self._vocab_size), dtype=np.float32)

        for b in range(batch):
            row_ids, row_mask = ids_np[b], mask_np[b]
            context_ids = [int(i) for i, m in zip(row_ids, row_mask) if not m and i != PAD_ID]
            offset = 0
            for t in range(seq_len):
                if not row_mask[t]:
                    continue
                token_id = int(row_ids[t])
                if token_id == PAD_ID:
                    raise ValueError(f"row {b}, position {t}: answer_mask marks a PAD_ID position")
                if not 0 <= token_id < self._vocab_size:
                    raise ValueError(f"row {b}, position {t}: token id {token_id} outside [0, {self._vocab_size})")
                key = self._encode_key(context_ids, offset)
                logits[b, t, :] = self._association_matrix @ key
                offset += 1

        return torch.from_numpy(logits)
