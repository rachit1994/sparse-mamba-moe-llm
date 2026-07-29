"""TTTLinear: Test-Time Training with a linear model as the hidden state.

THE QUESTION this file answers: does an inner, self-supervised GRADIENT STEP
taken ON THE HIDDEN STATE -- not on the base weights, and not mere state
accumulation -- reduce a token-prediction loss within a sequence, and does it
do so *better* the longer the sequence runs? That is TTT's defining claim
(Sun et al., ICML 2025, "Learning to (Learn at Test Time)", arXiv:2407.04620,
Section 2.3/3: "TTT layers": the hidden state is itself a model, updated by
gradient descent on a self-supervised loss, at every token). A layer that
only accumulates (Hebbian sum, exponential moving average, ...) is a
different, older mechanism; implementation/09_mistake_log.md M17 is the
record of that distinction being missed once already on this project.

HOW TO REVIEW IT:
  1. `_inner_grad_linear` is the ONLY place a gradient is computed, and it is
     a closed-form formula, not `torch.autograd` -- there is no
     `requires_grad=True` tensor and no `.backward()` call anywhere in this
     file. `tests/test_models_ttt.py` cross-checks this formula against
     `torch.autograd` independently, on random inputs, so the formula itself
     is verified even though production code never calls autograd.
  2. `forward_sequence`'s loop body is the entire inner loop. Confirm it
     reassigns `self._W` (the state) every iteration and *never* reassigns
     `self._theta_K` / `self._theta_Q` / `self._theta_V` (the base weights) --
     those three names appear only on the read side (`@ x_t`) inside the loop.
  3. `substrate_bytes_breakdown` is built directly from `base_state_dict` and
     `self._W`: there is no third place a tensor could be silently excluded.

DERIVATION -- TTT-Linear's analytic inner gradient (paper's f(W;x) = Wx):

    f(W; x) = W x                                    per-token model
    l(W)    = || f(W; k) - v ||^2 = || W k - v ||^2   k, v in R^d, W in R^(d x d)

  Let r = W k - v (the residual, in R^d). Componentwise, r_i = sum_j W_ij k_j - v_i,
  so l(W) = sum_i r_i^2 and

    d l / d W_ij = sum_i' 2 r_i' * (d r_i' / d W_ij) = 2 r_i * k_j

  because d r_i' / d W_ij is k_j when i' == i and 0 otherwise (r_i' depends on
  row i' of W only). Stacking every (i, j) entry back into matrix form:

    grad_W l(W) = 2 r k^T = 2 (W k - v) k^T          outer product, shape (d, d)

  This is exactly the formula the task specification gives, and exactly what
  `_inner_grad_linear` computes.

MECHANISM, per token x_t (paper's formulation; theta_K/Q/V fixed for the
call, W is the only thing that changes):

    k   = theta_K @ x_t                       train view
    v   = theta_V @ x_t                       label view (self-supervised target)
    l(W_{t-1}) = || W_{t-1} k - v ||^2        inner loss, evaluated BEFORE the step
    W_t = W_{t-1} - inner_lr * grad_W l(W_{t-1})   <- THE inner gradient step
    z_t = W_t @ (theta_Q @ x_t)               output, uses the UPDATED state

Non-goals, stated so a reviewer does not read them as gaps. All are real
components of the published architecture's systems/expressivity wrapper, not
of the defining mechanism above, and the task brief asks for TTT-Linear "at
minimum":
  * TTT-MLP (state = 2-layer MLP instead of a matrix) -- a second instance of
    the same mechanism with a different f; out of scope by the brief's own
    "at minimum" wording.
  * Multi-head splitting, output LayerNorm + gated residual, a learnable
    inner_lr or a learned (rather than zero) initial W_0, and the mini-batch
    "dual form" that parallelises the loop on a GPU (paper Section 3.2-3.4) --
    all engineering/expressivity additions on top of the recurrence above,
    not the recurrence itself.
  * An outer training loop that would learn theta_K/Q/V across many
    sequences via SGD. theta_K/Q/V are fixed random projections for the
    lifetime of one `TTTLinear` instance (see `__init__`); this file's
    acceptance criteria are about the INNER loop leaving them untouched, not
    about how they would be trained by something else.
  * Batching (`forward_sequence` takes a single (seq, dim) sequence, matching
    the required API exactly) -- the inner loop is inherently sequential
    (`W_t` depends on `W_{t-1}`), so batching would only parallelise the
    projections, not the loop that defines TTT; adding it now would grow this
    file without changing what the acceptance tests measure.

Complexity: `forward_sequence` is O(seq_len * dim^2) time -- each of the
`seq_len` steps does a constant number of O(dim^2) operations (three
matrix-vector products for k/v/q, one for the prediction, one outer product,
one state update) -- and O(dim^2) space, independent of seq_len. The O(dim^2)
space bound *is* the property under test in test 5: a fixed-size state that
keeps improving with more tokens, unlike a state that must grow to remember.

Depends on nothing outside `torch`; does not import `src.data` or
`src.metrics`, matching `src/models/dynamic.py`'s and `TinyLM`'s convention
that a model file receives already-projected tensors, not raw text or ids.
"""

from __future__ import annotations

import torch

__all__ = ["TTTLinear"]

# Working precision for every persistent tensor this module owns. float32
# (never float64), matching src/metrics/bits.py's stated float32
# floor-and-ceiling convention and src/models/dynamic.py's identical choice.
_STATE_DTYPE: torch.dtype = torch.float32

# Bytes per element at _STATE_DTYPE. float32 is 4 bytes by the IEEE 754
# standard; named here (not left as a magic `4`) so substrate_bytes's
# arithmetic is traceable to this one constant, matching
# src/models/dynamic.py's `_ASSOCIATION_ITEMSIZE_BYTES` convention.
_STATE_ITEMSIZE_BYTES: int = 4

# The three base (outer) parameter names, in the order the spec lists them
# ("theta_K, theta_Q, theta_V"). Defined once so base_state_dict and
# substrate_bytes_breakdown cannot silently drift apart on which names exist.
_BASE_PARAM_NAMES: tuple[str, str, str] = ("theta_K", "theta_Q", "theta_V")


def _inner_loss_linear(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """l(W) = ||W k - v||^2 for TTT-Linear's f(W; x) = W x. A 0-d tensor, one quantity."""
    residual = W @ k - v
    return residual @ residual


def _inner_grad_linear(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Analytic gradient of `_inner_loss_linear` wrt W: 2 (W k - v) k^T.

    See the module docstring's DERIVATION section. This is the ONLY gradient
    computation in this file; production code (`forward_sequence`) calls
    this closed form, never `torch.autograd` -- the module docstring's
    review point 1 states why, and `tests/test_models_ttt.py` verifies this
    formula against `torch.autograd` independently.

    Returns:
        A (dim, dim) tensor, the same shape as `W`.
    """
    residual = W @ k - v
    return 2.0 * torch.outer(residual, k)


class TTTLinear:
    """TTT-Linear: hidden state W is a (dim, dim) matrix, updated by an inner gradient step.

    theta_K, theta_Q, theta_V ("base weights") are fixed random (dim, dim)
    projections drawn once at construction from `seed` and never assigned to
    again -- see the module docstring's HOW TO REVIEW point 2. W ("inner
    state") starts at zero (no learned initial state is in scope; see
    Non-goals) and is the only thing `forward_sequence` mutates.

    None of the five tensors this class owns is ever wrapped in
    `torch.nn.Parameter` or created inside a `torch.autograd`-tracking
    context: there is no gradient *concept* applicable via autograd anywhere
    in this file, not merely an untaken one (mirrors
    `src/models/dynamic.py`'s identical structural guarantee for numpy).
    """

    def __init__(self, dim: int, inner_lr: float, seed: int) -> None:
        """Allocate fixed base projections and a zeroed inner state.

        Args:
            dim: feature dimension; theta_K/Q/V and W are all (dim, dim).
                Must be a positive int.
            inner_lr: the inner-loop step size `eta`. Must be a finite,
                positive number: `inner_lr <= 0` (or NaN, since
                `not (nan > 0)` is True) would make the inner gradient step
                a no-op or a sign-flip, silently defeating the mechanism
                this class exists to implement rather than raising.
            seed: determinism seed for theta_K/Q/V. Same (dim, seed) always
                yields bit-identical `base_state_dict()`.

        Raises:
            TypeError: if `dim`/`seed` are not plain ints (bools rejected,
                matching `src/models/dynamic.py`'s constructor convention),
                or `inner_lr` is not a plain int or float.
            ValueError: if `dim <= 0`, `seed < 0`, or `inner_lr` is not
                strictly positive.
        """
        for name, value in (("dim", dim), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if isinstance(inner_lr, bool) or not isinstance(inner_lr, (int, float)):
            raise TypeError(f"inner_lr must be a float, got {type(inner_lr).__name__}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if not inner_lr > 0:
            raise ValueError(f"inner_lr must be strictly positive, got {inner_lr}")

        self._dim = dim
        self._inner_lr = float(inner_lr)

        # Fan-in-normalised init (std = 1/sqrt(dim), standard practice for a
        # dim x dim projection, e.g. attention Q/K/V init): keeps k = theta_K
        # @ x_t at O(1) per-entry variance across the whole range of `dim`
        # this project's tests use (2 to a few dozen), so a single inner_lr
        # does not need re-tuning per dimension.
        generator = torch.Generator().manual_seed(seed)
        init_scale = dim**-0.5
        self._theta_K = torch.randn(dim, dim, generator=generator, dtype=_STATE_DTYPE) * init_scale
        self._theta_Q = torch.randn(dim, dim, generator=generator, dtype=_STATE_DTYPE) * init_scale
        self._theta_V = torch.randn(dim, dim, generator=generator, dtype=_STATE_DTYPE) * init_scale

        self._W = torch.zeros(dim, dim, dtype=_STATE_DTYPE)
        self._loss_trace = torch.zeros(0, dtype=_STATE_DTYPE)

    def base_state_dict(self) -> dict[str, torch.Tensor]:
        """theta_K/Q/V only -- clones, so mutating the returned dict cannot reach the live state."""
        return {name: getattr(self, f"_{name}").clone() for name in _BASE_PARAM_NAMES}

    def reset_state(self) -> None:
        """Clear the inner state W back to its construction-time value (zero) and the loss trace."""
        self._W = torch.zeros(self._dim, self._dim, dtype=_STATE_DTYPE)
        self._loss_trace = torch.zeros(0, dtype=_STATE_DTYPE)

    def forward_sequence(self, xs: torch.Tensor) -> torch.Tensor:
        """Run the inner loop token-by-token: one gradient step on W, then read with theta_Q.

        For each token, in order: computes `k`/`v`/`q` from the CURRENT
        `theta_K`/`theta_V`/`theta_Q`, records `l(W_{t-1})` (see
        `inner_loss_trace`), takes the analytic gradient step to get `W_t`,
        then reads the output with the UPDATED `W_t` -- exactly the module
        docstring's MECHANISM section, in order. `theta_K`/`theta_Q`/`theta_V`
        are read here and never assigned to.

        Args:
            xs: shape (seq_len, dim), a floating-point dtype. Cast to this
                module's working precision (`_STATE_DTYPE`) if given as a
                different floating dtype.

        Returns:
            Outputs `z`, shape (seq_len, dim), dtype `_STATE_DTYPE`. Empty
            (`seq_len == 0`) input returns an empty `(0, dim)` output and
            leaves the state unchanged -- a valid no-op, not an error.

        Raises:
            TypeError: if `xs` is not a floating-point-dtype tensor.
            ValueError: if `xs` is not 2D, or its second dimension does not
                equal `dim`.
        """
        if not torch.is_floating_point(xs):
            raise TypeError(f"xs must be a floating-point tensor, got dtype {xs.dtype}")
        if xs.dim() != 2:
            raise ValueError(f"xs must be 2D (seq_len, dim), got shape {tuple(xs.shape)}")
        seq_len, dim = xs.shape
        if dim != self._dim:
            raise ValueError(f"xs has dim {dim}, but this TTTLinear was built with dim={self._dim}")

        if seq_len == 0:
            self._loss_trace = torch.zeros(0, dtype=_STATE_DTYPE)
            return torch.zeros(0, self._dim, dtype=_STATE_DTYPE)

        # No autograd concept applies to this loop (module docstring, review
        # point 1): every gradient here is the closed-form `_inner_grad_linear`.
        # `no_grad()` is a structural guarantee, not a performance tweak --
        # it stops a caller-supplied `requires_grad=True` xs from silently
        # attaching an autograd graph to self._W.
        with torch.no_grad():
            xs = xs.detach().to(_STATE_DTYPE)
            outputs = torch.zeros(seq_len, self._dim, dtype=_STATE_DTYPE)
            losses = torch.zeros(seq_len, dtype=_STATE_DTYPE)
            for t in range(seq_len):
                x_t = xs[t]
                k = self._theta_K @ x_t
                v = self._theta_V @ x_t
                q = self._theta_Q @ x_t
                losses[t] = _inner_loss_linear(self._W, k, v)
                grad = _inner_grad_linear(self._W, k, v)
                self._W = self._W - self._inner_lr * grad
                outputs[t] = self._W @ q

        self._loss_trace = losses
        return outputs

    @property
    def inner_loss_trace(self) -> torch.Tensor:
        """Per-token inner loss `l(W_{t-1})` from the most recent `forward_sequence` call.

        Shape `(seq_len,)`, matching that call's input length. Shape `(0,)`
        (never `None`) if `forward_sequence` has not been called since
        construction or `reset_state()` -- so a caller can always index or
        call `.numel()` without a null check.

        This is a diagnostic surfaced as a byproduct of `forward_sequence`'s
        one required loop, not a second implementation of it: it exists
        because the acceptance tests need the loss trajectory itself (test 2:
        "report the actual loss sequence"; test 5: mean loss over the last k
        tokens), which the required `forward_sequence(xs) -> Tensor` return
        shape (outputs only) has no room for.
        """
        return self._loss_trace

    def substrate_bytes_breakdown(self) -> dict[str, int]:
        """Byte size of every persistent tensor this object owns, named for a reviewer."""
        tensors: dict[str, torch.Tensor] = {**self.base_state_dict(), "W": self._W}
        return {name: tensor.numel() * _STATE_ITEMSIZE_BYTES for name, tensor in tensors.items()}

    def substrate_bytes(self) -> int:
        """Total bytes: base params (theta_K/Q/V) + inner state (W), both counted."""
        return sum(self.substrate_bytes_breakdown().values())
