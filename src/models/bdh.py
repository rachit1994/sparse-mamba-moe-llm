"""BDH: a neuron-synapse layer whose working memory is a Hebbian synapse matrix.

THE QUESTION this file answers: can a "linear attention as a recurrent
Hebbian synapse" formulation -- the BDH-GPU view of "The Dragon Hatchling"
(Pathway, Sept 2025, arXiv:2509.26507) -- produce attention-LIKE retrieval
(one token driving the representation of another it was paired with) out of
a purely LOCAL update rule, with no softmax and no learned weights, and with
every unit of adaptive state accounted for? implementation/09_mistake_log.md
M17: a Hebbian outer-product memory alone is NOT BDH ("no neuron-synapse
dynamics or emergent attention") -- this file exists to be the thing M17
says was missing. initial_research/10_dynamic_substrate.md section 3.1 names
BDH's two defining claims: "working memory during inference relies entirely
on synaptic plasticity with Hebbian learning" and "attention emerges from
pairwise synapse updates governed by local correlation."

HOW TO REVIEW IT:
  1. `base_state_dict()` returns exactly the two tensors (`encoder`,
     `decoder`) that must be bit-identical before and after any sequence of
     `forward_sequence()` calls -- confirm neither is ever assigned to
     outside `__init__`. All adaptation lives in `self._sigma`, mutated in
     exactly one place: the marked "HEBBIAN UPDATE" line in
     `forward_sequence`.
  2. `neuron_activations()` is a PURE function of its input (it never reads
     or writes `self._sigma`) -- this is what lets tests predict exact
     synapse deltas from public tensors alone (see MECHANISM below).
  3. `substrate_bytes()` is built from `substrate_bytes_breakdown()`, which
     is built directly from `base_state_dict()` and `synapse_state()` --
     there is no third place a tensor could be silently excluded from the
     count (mistake log guidance: "under-counting persistent state is how a
     dynamic arm fakes a win").

MECHANISM (BDH-GPU, linear-attention-equivalent form). Per token t, with
`h_t` the token's dim-dimensional input (`xs[t]`, used directly -- see
Non-goals):

    y_t     = top_k(relu(encoder^T @ h_t))          # sparse, positive: neuron_activations()
    sigma_t = decay * sigma_{t-1} + outer(y_t, y_{t-1})   # HEBBIAN, local
    a_t     = sigma_t @ y_t                          # retrieval through the synapse
    out_t   = (y_t + a_t) @ decoder                  # forward_sequence()'s t-th row

`encoder` (dim x n_neurons) and `decoder` (n_neurons x dim) are the calling
brief's E and D: fixed at construction, never updated, both inherently
low-rank (rank <= dim, since dim << n_neurons).

DIVERGENCE FROM A LITERAL SAME-TIMESTEP READING (flagged per this task's own
"if you believe the spec misstates the paper, say so before building" rule,
since arXiv is unreachable here to check directly): the calling brief writes
`outer(y_t, y_t_key)` with both terms indexed at the SAME step t, and lists
"a low-rank read/write pair" as fixed params beyond E and D. Taken literally
-- y_t_key a second projection of the SAME h_t, with h_t itself defined by
nothing but the current external input -- sigma could only ever accumulate a
token's correlation with ITSELF: every (value, key) pair written at step t
is a function of that same step's input alone, so sigma can never learn that
token A is followed by token B, no matter how many extra weight matrices
project h_t into key/query space. That would make Property 3 (attention
emerging from local correlation) unsatisfiable by construction, and this
file's Test 5 (emergent association, "the property that matters") would
report a ratio of ~1 regardless of tuning -- a mathematical fact about the
literal reading, not a bug to be tuned around.

This implementation makes the minimal correction: the Hebbian KEY is the
PREVIOUS step's activation, `y_{t-1}` (zero at t=0), not a same-step term.
This is still local (adjacent in time, not a global/softmax comparison
against the whole sequence), still Hebbian (co-activity of two neuron
populations strengthens the synapse between them), and is the "pre precedes
post" pairing that spike-timing-dependent plasticity -- the paper's own
neuroscience framing -- names. It is also exactly what makes the calling
brief's own Test 2 wording work: "a pair of TOKENS that co-activate neurons
i and j" reads most naturally as two adjacent tokens, not one token's
self-correlation.

The read side reuses `encoder`'s OWN output as the query (`a_t = sigma_t @
y_t`, not a separate projection): retrieval must land on the neuron-index
columns that were previously written as keys, which is only guaranteed when
key and query are produced by the same low-rank map. An independent extra
"read" projection was considered and rejected: two independently-seeded
(dim, n_neurons) matrices produce statistically uncorrelated top-k supports
(expected column overlap ~ (top_k / n_neurons) per shared index, negligible
at the n_neurons=256, top_k<=32 scale this file's tests use), so a retrieval
query drawn from an independent matrix would, in expectation, land on
columns sigma never wrote -- breaking Property 3 for the same reason the
literal same-timestep reading does. The brief's "low-rank read/write pair"
is realized here as `encoder` used in its two temporal roles (this step's
query, next step's key) rather than as extra tensors whose only effect would
be to break the property they are meant to serve.

Non-goals (deliberate, stated so a reviewer does not read them as gaps):
  * No hidden state carried through `h_t` itself across positions --
    `h_t = xs[t]` exactly, every step. All cross-token memory lives in
    `self._sigma`, matching Property 1 ("entirely in synaptic state") with
    no second, unaudited channel (a residual `h`-carry) through which
    information could also leak.
  * No batch dimension: `forward_sequence` is `(seq, dim) -> (seq, dim)`,
    one sequence, matching the calling brief's literal signature. Stacking
    independent sequences is a caller concern, not this layer's.
  * No multi-layer stacking, no token embedding/un-embedding, no softmax
    anywhere -- a softmax attention block would BE the thing this file
    exists to avoid building (calling brief, Property 3).
  * `top_k` selection can never fabricate positive mass ReLU has already
    zeroed: an all-non-positive-pre-activation row yields fewer than
    `top_k` nonzero entries. This is correct, not a bug (edge-case ladder
    rung 1) -- see `_sparsify_top_k`.

Depends on nothing outside `torch`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["BDHLayer"]

# Fan-in scaling (LeCun/Xavier-style) for the two fixed random projections:
# keeps `encoder^T @ h_t`'s pre-activation variance near 1 for unit-scale
# `h_t`, so a typical row has a healthy mix of positive and negative entries
# for `top_k` to select among (rather than saturating all-positive or
# all-negative). Not a paper-sourced constant -- arXiv:2509.26507 does not
# specify an initialization scheme for these fixed projections, and this is
# a standard, reasonable default, flagged as such rather than presented as
# derived from the paper.
def _fan_in_std(fan_in: int) -> float:
    """Standard-deviation for a fixed random projection with `fan_in` inputs."""
    return 1.0 / math.sqrt(fan_in)


def _sparsify_top_k(activations: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero every entry except the `top_k` largest per row of a non-negative tensor.

    This is Property 2's enforcement point: "sparsity is not incidental."
    `activations` must already be ReLU'd -- this function only selects a
    rank-based mask, it never changes a value's sign or magnitude.

    Args:
        activations: non-negative tensor, shape `(..., n)`.
        top_k: number of entries to keep per row.

    Returns:
        Same shape as `activations`, entries outside the per-row top `top_k`
        set to exactly `0.0`. If fewer than `top_k` entries in a row are
        strictly positive (e.g. an all-zero row), fewer than `top_k` survive
        as nonzero -- `top_k` is a selection RANK, not a floor that can
        manufacture activation ReLU has already removed (edge-case ladder
        rung 1).

    Raises:
        ValueError: if `top_k` is not in `[1, activations.shape[-1]]`.
    """
    n = activations.shape[-1]
    if not 1 <= top_k <= n:
        raise ValueError(f"top_k must be in [1, {n}], got {top_k}")
    _, top_indices = torch.topk(activations, k=top_k, dim=-1)
    keep_mask = torch.zeros_like(activations, dtype=torch.bool)
    keep_mask.scatter_(-1, top_indices, True)
    return torch.where(keep_mask, activations, torch.zeros((), dtype=activations.dtype))


class BDHLayer:
    """A single BDH-GPU layer: sparse positive neurons, Hebbian synaptic working memory.

    Not an `nn.Module`: `encoder` and `decoder` are fixed at construction and
    never receive a gradient step (there is no gradient *concept* applicable
    to them, matching `src/models/dynamic.py`'s "no gradient, ever" posture
    for its own fixed projections) -- all adaptation is the Hebbian update to
    `self._sigma`, plain tensor arithmetic with `torch.no_grad` semantics by
    construction (nothing here is ever wrapped in `requires_grad_(True)`).

    Complexity: `forward_sequence` is O(seq_len * n_neurons^2) time (one
    (n_neurons x n_neurons) outer product and one matrix-vector product per
    token) and O(n_neurons^2) space for `sigma` (dominant over the
    O(dim * n_neurons) `encoder`/`decoder`, since n_neurons >> dim by
    Property 2). At this file's test scale (n_neurons <= 256, seq_len in the
    low hundreds) that is at most ~10^7 multiply-adds per call -- well under
    a second on one CPU core, so no chunked/vectorized-scan rewrite is
    justified (code style: no speculative optimisation for a scale the
    tests do not exercise).
    """

    def __init__(self, dim: int, n_neurons: int, top_k: int, decay: float, seed: int) -> None:
        """Allocate the fixed projections and a zeroed synapse.

        Args:
            dim: width of the token representation `forward_sequence`
                consumes and produces. Must be a positive int.
            n_neurons: size of the sparse neuron population (the calling
                brief's `n`; `n_neurons >> dim` is the paper's
                "high-dimensional" property but is not itself enforced here
                -- see the class docstring). Must be a positive int.
            top_k: number of neurons kept active per token, `1 <= top_k <=
                n_neurons`.
            decay: per-token multiplicative forgetting factor applied to
                `sigma` before each token's write, `0.0 <= decay <= 1.0`.
                `decay == 1.0` means no forgetting; `decay < 1.0` is what
                Test 6 exercises.
            seed: determinism seed for `encoder`/`decoder`. Same (dim,
                n_neurons, seed) always yields bit-identical
                `base_state_dict()`.

        Raises:
            TypeError: if `dim`/`n_neurons`/`top_k`/`seed` is not a plain
                int (bools rejected), or `decay` is not a plain float/int.
            ValueError: if `dim < 1`, `n_neurons < 1`, `top_k` outside
                `[1, n_neurons]`, `decay` outside `[0.0, 1.0]`, or
                `seed < 0`.
        """
        for name, value in (("dim", dim), ("n_neurons", n_neurons), ("top_k", top_k), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError(f"decay must be a float, got {type(decay).__name__}")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if not 1 <= top_k <= n_neurons:
            raise ValueError(f"top_k must be in [1, {n_neurons}], got {top_k}")
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"decay must be in [0.0, 1.0], got {decay}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        self.dim = dim
        self.n_neurons = n_neurons
        self.top_k = top_k
        self.decay = float(decay)

        generator = torch.Generator().manual_seed(seed)
        self._encoder = torch.randn(dim, n_neurons, generator=generator, dtype=torch.float32)
        self._encoder *= _fan_in_std(dim)
        self._decoder = torch.randn(n_neurons, dim, generator=generator, dtype=torch.float32)
        self._decoder *= _fan_in_std(n_neurons)
        self.reset_state()

    def _validate_xs(self, xs: torch.Tensor) -> None:
        """Shared shape/dtype contract for `forward_sequence` and `neuron_activations`.

        Raises:
            TypeError: if `xs` is not a floating-point `torch.Tensor`.
            ValueError: if `xs` is not 2D `(seq_len, dim)`, `seq_len == 0`,
                or the trailing dimension does not equal `self.dim`.
        """
        if not torch.is_tensor(xs):
            raise TypeError(f"xs must be a torch.Tensor, got {type(xs).__name__}")
        if not torch.is_floating_point(xs):
            raise TypeError(f"xs must be a floating-point tensor, got dtype {xs.dtype}")
        if xs.dim() != 2:
            raise ValueError(f"xs must be 2D (seq_len, dim), got shape {tuple(xs.shape)}")
        seq_len, width = xs.shape
        if seq_len == 0:
            raise ValueError("xs has seq_len 0; nothing to process")
        if width != self.dim:
            raise ValueError(f"xs has trailing dim {width}, expected dim={self.dim}")

    def base_state_dict(self) -> dict[str, torch.Tensor]:
        """The fixed params `forward_sequence` must never modify (the gate's hash target)."""
        return {"encoder": self._encoder, "decoder": self._decoder}

    def reset_state(self) -> None:
        """Clear the synapse: `sigma <- 0`. The only thing this resets; base params are untouched."""
        self._sigma = torch.zeros((self.n_neurons, self.n_neurons), dtype=torch.float32)

    def synapse_state(self) -> torch.Tensor:
        """The live `sigma` tensor (working memory). Callers wanting a snapshot must `.clone()` it."""
        return self._sigma

    def neuron_activations(self, xs: torch.Tensor) -> torch.Tensor:
        """Sparse positive neuron activations `y_t` for every position, independent of `sigma`.

        Pure function of `xs` alone (never reads or writes `self._sigma`):
        `y_t = top_k(relu(encoder^T @ xs[t]))`. This is what `forward_sequence`
        uses internally as value, key (one step later), and query -- exposing
        it separately lets a caller (or test) predict the exact Hebbian
        update `forward_sequence` will make without duplicating its loop.

        Args:
            xs: token representations, shape `(seq_len, dim)`, floating dtype.

        Returns:
            `(seq_len, n_neurons)` tensor, entrywise `>= 0`, with exactly
            `top_k` nonzero entries per row for any row with at least
            `top_k` strictly positive pre-activations (see
            `_sparsify_top_k` for the degenerate-row exception).

        Raises:
            TypeError: if `xs` is not a floating-point tensor.
            ValueError: if `xs` is not 2D, `seq_len == 0`, or its trailing
                dimension does not equal `self.dim`.
        """
        self._validate_xs(xs)
        pre_activation = xs.to(torch.float32) @ self._encoder
        return _sparsify_top_k(F.relu(pre_activation), self.top_k)

    def forward_sequence(self, xs: torch.Tensor) -> torch.Tensor:
        """Run the layer over one sequence, mutating `sigma` one token at a time.

        Per position t (see module MECHANISM for the full derivation and the
        documented divergence from a literal same-timestep reading):

            sigma_t = decay * sigma_{t-1} + outer(y_t, y_{t-1})   # y_{-1} := 0
            out_t   = (y_t + sigma_t @ y_t) @ decoder

        `sigma` is `self._sigma`: it PERSISTS across separate
        `forward_sequence` calls until `reset_state()` clears it (Test 4's
        "state is load-bearing" requirement) -- this call continues from
        whatever state the layer is already in, it does not start fresh.

        Args:
            xs: token representations, shape `(seq_len, dim)`, floating dtype.

        Returns:
            `(seq_len, dim)` tensor, `out_t` at each row.

        Raises:
            TypeError: if `xs` is not a floating-point tensor.
            ValueError: if `xs` is not 2D, `seq_len == 0`, or its trailing
                dimension does not equal `self.dim`.
        """
        self._validate_xs(xs)
        y = self.neuron_activations(xs)  # (seq_len, n_neurons); pure, computed once up front
        seq_len = xs.shape[0]
        outputs = torch.empty((seq_len, self.dim), dtype=torch.float32)
        y_previous = torch.zeros(self.n_neurons, dtype=torch.float32)
        for t in range(seq_len):
            y_t = y[t]
            # HEBBIAN UPDATE -- the ONLY line in this file that assigns to
            # self._sigma. Key = previous step's activation (see module
            # MECHANISM for why); value = this step's activation. Neurons
            # that fired last step and neurons firing now have their
            # synapse strengthened by exactly their co-activation product.
            self._sigma = self.decay * self._sigma + torch.outer(y_t, y_previous)
            retrieved_t = self._sigma @ y_t  # query reuses y_t -- see MECHANISM
            outputs[t] = (y_t + retrieved_t) @ self._decoder
            y_previous = y_t
        return outputs

    def substrate_bytes_breakdown(self) -> dict[str, int]:
        """Byte size of every persistent tensor this layer owns, named for a reviewer.

        Built directly from `base_state_dict()` and `synapse_state()` -- see
        module docstring point 3: there is no other accounting path to fall
        out of sync with.
        """
        tensors = {**self.base_state_dict(), "synapse": self.synapse_state()}
        return {name: tensor.numel() * tensor.element_size() for name, tensor in tensors.items()}

    def substrate_bytes(self) -> int:
        """Total persistent bytes: fixed params (`encoder`, `decoder`) PLUS `sigma`.

        `sigma` is `n_neurons x n_neurons` and dominates whenever
        `n_neurons >> dim` (Property 2) -- see `substrate_bytes_breakdown`
        for the per-tensor split Test 7 checks against an independent hand
        count.
        """
        return sum(self.substrate_bytes_breakdown().values())
