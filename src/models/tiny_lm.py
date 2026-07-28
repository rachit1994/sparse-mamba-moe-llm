"""TinyLM: the control-arm decoder-only transformer.

This arm exists to reproduce the published dense-transformer baseline of
2.0 bits/param (Allen-Zhu & Li) -- see implementation/00_START_HERE.md
section 1 and implementation/06_agent_briefs/B3_baseline_model.md. It is
deliberately "boring": standard pre-norm causal self-attention, learned
absolute position embeddings, GELU MLP, optional weight tying. No novelty
here is the point -- novelty belongs to the dynamic-substrate arm this one
is compared against, and if this arm is wrong the comparison is worthless.

Every architectural choice made here (bias on/off, MLP ratio, tying) is
mirrored exactly by ``src.models.config.analytic_param_count`` -- the two
must never drift apart; ``tests/test_models_tiny_lm.py`` asserts they agree
for every entry in ``PRESET_CONFIGS``.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig

__all__ = [
    "TinyLM",
    "causal_lm_shift",
    "truncate_ids",
]


class _CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention, computed without fused SDPA kernels.

    Deliberately implemented as explicit matmul + softmax rather than
    ``torch.nn.functional.scaled_dot_product_attention``: SDPA can select
    between multiple numerical backends (math / flash / mem-efficient), and
    which one it picks is not guaranteed stable across processes on every
    platform. This project's determinism requirement (R3: same seed ->
    bit-identical weights) needs every op in the forward pass to take one,
    predictable numerical path (implementation/02_testing_philosophy.md
    section 4; implementation/03_edge_cases_and_scares.md H3).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = self.causal_mask[:seq_len, :seq_len]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.proj(y)


class _MLP(nn.Module):
    """Standard 4x-expansion GELU MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """One pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = _CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    """Decoder-only causal transformer language model.

    Architecture (see module docstring for the "why boring"): token
    embedding + learned absolute positional embedding -> `n_layer` pre-norm
    transformer blocks -> final layernorm -> linear output head, optionally
    weight-tied to the token embedding.

    Both `wte` and `wpe` are `nn.Embedding` modules (not raw `nn.Parameter`s)
    specifically so that `src.metrics.bits.count_parameters`'s *structural*
    embedding detection (``isinstance(module, nn.Embedding)``) finds both of
    them -- stated explicitly because it is load-bearing for the
    `include_embeddings=False` convention (implementation/03_edge_cases_and_scares.md
    S3): with this architecture, `include_embeddings=False` excludes *both*
    the token and positional embedding tables, but never `lm_head` even when
    untied (`lm_head` is an `nn.Linear`, not an `nn.Embedding`).

    Attributes:
        config: the `ModelConfig` this instance was built from.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Weight tying by shared Parameter *object* identity (not just
            # equal values): `named_parameters(remove_duplicate=True)` --
            # torch's default, and what src.metrics.bits.count_parameters
            # relies on -- dedups by `id()`, so this is what makes a tied
            # pair count once rather than twice (implementation/03_edge_cases_and_scares.md S3,
            # src/models/config.py module docstring).
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the model, returning next-token logits at every position.

        ``logits[..., t, :]`` is the model's prediction for the token that
        *follows* ``idx[..., t]`` (standard causal-LM convention: it is
        conditioned on ``idx[..., :t+1]`` only). Callers that need
        ``(logits, targets, answer_mask)`` aligned so that
        ``logits[..., t, :]`` predicts ``targets[..., t]`` (the contract
        required by ``src.metrics.bits.answer_cross_entropy_bits``) must
        shift by one -- see :func:`causal_lm_shift`.

        Args:
            idx: token ids, shape ``(batch, seq_len)``, dtype ``torch.long``.

        Returns:
            Logits, shape ``(batch, seq_len, vocab_size)``, dtype
            ``torch.float32``.

        Raises:
            TypeError: if ``idx`` is not an integer-dtype tensor.
            ValueError: if ``idx`` is not 2D, if ``batch == 0`` or
                ``seq_len == 0``, if ``seq_len > config.block_size`` (this
                model never silently truncates -- callers must truncate
                explicitly and log it, see :func:`truncate_ids`), or if any
                token id is outside ``[0, vocab_size)``.
        """
        if torch.is_floating_point(idx) or idx.dtype == torch.bool:
            raise TypeError(f"idx must be an integer-dtype tensor of token ids, got {idx.dtype}")
        if idx.dim() != 2:
            raise ValueError(f"idx must be 2D (batch, seq_len), got shape {tuple(idx.shape)}")
        batch, seq_len = idx.shape
        if batch == 0:
            raise ValueError("idx has batch size 0; nothing to run the model on")
        if seq_len == 0:
            raise ValueError("idx has seq_len 0; nothing to run the model on")
        if seq_len > self.config.block_size:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds block_size ({self.config.block_size}); "
                f"this model never silently truncates -- truncate explicitly with "
                f"truncate_ids() before calling forward()"
            )
        if idx.numel() > 0:
            min_id = int(idx.min())
            max_id = int(idx.max())
            if min_id < 0 or max_id >= self.config.vocab_size:
                raise ValueError(
                    f"idx contains token id(s) outside [0, {self.config.vocab_size}): "
                    f"min={min_id}, max={max_id}"
                )

        tok_emb = self.wte(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def causal_lm_shift(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align `TinyLM.forward` output with the `answer_cross_entropy_bits` contract.

    This is requirement R5's "logits hook"
    (implementation/06_agent_briefs/B3_baseline_model.md): a pure function
    with no side effects that turns raw model output plus the caller's
    prompt/answer bookkeeping into exactly the ``(logits, targets,
    answer_mask)`` triple ``src.metrics.bits.answer_cross_entropy_bits``
    expects -- see that function's docstring for the shape/dtype contract it
    enforces on its inputs, which this function's outputs satisfy by
    construction.

    ``TinyLM.forward`` follows the standard causal-LM convention:
    ``logits[..., t, :]`` predicts ``input_ids[..., t+1]``, not
    ``input_ids[..., t]``. ``answer_cross_entropy_bits`` requires the
    opposite alignment (``logits[..., t, :]`` already lined up with
    ``targets[..., t]``), so this function performs the shift-by-one:

    Args:
        logits: raw `TinyLM.forward` output, shape
            ``(..., seq_len, vocab_size)``.
        input_ids: the token ids that were fed to the model, shape
            ``(..., seq_len)``, integer dtype.
        answer_mask: boolean mask over *input_ids* positions, shape
            ``(..., seq_len)``. ``answer_mask[..., t] == True`` means
            ``input_ids[..., t]`` is itself an answer token (i.e. a token to
            be scored as a *target* once shifted).

    Returns:
        ``(shifted_logits, targets, shifted_answer_mask)``:

        * ``shifted_logits``: ``logits[..., :-1, :]``, shape
          ``(..., seq_len - 1, vocab_size)``.
        * ``targets``: ``input_ids[..., 1:]``, shape ``(..., seq_len - 1)``.
        * ``shifted_answer_mask``: ``answer_mask[..., 1:]``, shape
          ``(..., seq_len - 1)`` -- so ``shifted_answer_mask[..., t]``
          correctly marks whether ``targets[..., t]`` (== the *original*
          ``input_ids[..., t + 1]``) is an answer token.

    Raises:
        ValueError: if ``logits`` has fewer than 2 dims; if
            ``input_ids.shape != answer_mask.shape``; if
            ``logits.shape[:-1] != input_ids.shape``; or if the shared
            ``seq_len`` is less than 2 (nothing to shift).
    """
    if logits.dim() < 2:
        raise ValueError(
            f"logits must have at least 2 dims (..., vocab_size), got shape {tuple(logits.shape)}"
        )
    if input_ids.shape != answer_mask.shape:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} != answer_mask shape {tuple(answer_mask.shape)}"
        )
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} incompatible with input_ids shape "
            f"{tuple(input_ids.shape)}: expected logits.shape[:-1] == input_ids.shape"
        )
    seq_len = input_ids.shape[-1]
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2 to shift by one, got {seq_len}")

    shifted_logits = logits[..., :-1, :]
    targets = input_ids[..., 1:]
    shifted_answer_mask = answer_mask[..., 1:]
    return shifted_logits, targets, shifted_answer_mask


def truncate_ids(ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """Explicitly truncate a token-id sequence to `block_size`, logging if it does.

    `TinyLM.forward` refuses (raises) rather than silently truncating an
    over-length sequence (implementation/03_edge_cases_and_scares.md,
    Training / Huge: "truncate explicitly and log, never wrap silently").
    This is the explicit, logged truncation step callers are expected to
    apply *before* calling `forward` on real (variable-length) text.

    Args:
        ids: token ids, shape ``(..., seq_len)``.
        block_size: maximum length to keep. Must be a positive int.

    Returns:
        ``ids`` unchanged if ``seq_len <= block_size``, else
        ``ids[..., -block_size:]`` (keeps the most recent `block_size`
        tokens, standard causal-LM convention -- the discarded prefix can
        never be attended to again anyway).

    Raises:
        ValueError: if ``block_size <= 0``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    seq_len = ids.shape[-1]
    if seq_len <= block_size:
        return ids
    warnings.warn(
        f"truncate_ids: sequence length {seq_len} exceeds block_size {block_size}; "
        f"dropping the oldest {seq_len - block_size} token(s)",
        stacklevel=2,
    )
    return ids[..., -block_size:]
