"""End-to-end P0 pipeline: dataset -> tokenizer -> TinyLM -> bits-recovered metric.

Closes the integration gap identified in ``reports/001_P0_harness.md``: the three
P0 components (dataset generator, tokenizer, model/training loop, bits metric)
were individually verified but had never been composed, so the P0 controls
(null model, unseen-person leakage) could not actually be run. This module is
the composition; ``tests/test_integration_p0.py`` is where those controls
finally execute.

Pipeline, in order, all driven by one ``(n_people, seed)`` pair:

1. :func:`src.data.generate.write_dataset` -- generates train/probe/unseen_person/
   sealed splits to a scratch directory (a fresh ``tempfile.TemporaryDirectory``
   per call, so runs never collide and nothing is left on disk).
2. :func:`src.data.tokenizer.build_tokenizer` -- the closed, word-level
   vocabulary (``vocab_size == 2818``).
3. Training text is tokenized and padded (never truncated -- see
   :class:`TrainingTextTooLongError`) to ``config.block_size``, then handed to
   :func:`src.train.train_loop` as its ``data`` argument. ``steps=0`` runs the
   identical construction path (seeded model init, optimizer, batcher) with
   zero optimizer steps -- this is what makes it a legitimate **null-model**
   control (implementation/02_testing_philosophy.md section 2, C1): the null
   model is not a different code path, it is this same pipeline stopped before
   the first gradient step.
4. Each (person, attribute) probe/unseen-person record is encoded with
   :meth:`src.data.tokenizer.Tokenizer.encode_qa`, run through the model, shifted
   with :func:`src.models.tiny_lm.causal_lm_shift`, and scored with
   :func:`src.metrics.bits.answer_cross_entropy_bits`.
5. :func:`src.metrics.bits.bits_recovered_per_attribute` aggregates, with its own
   live per-item entropy-bound check; :func:`run_pipeline` adds a second,
   coarser live check at the total-bits level against
   ``schema.dataset_entropy_bits(n_people_scored)`` (implementation/04_golden_dataset.md
   section 4: "bits_recovered <= dataset entropy" must be live in code, not only
   in tests).

Scope note (S6, implementation/03_edge_cases_and_scares.md): ``eval_split`` is
deliberately restricted to ``Split.PROBE`` and ``Split.UNSEEN_PERSON``.
``Split.SEALED`` is intentionally NOT reachable from here -- the sealed split
may be looked at only once per phase gate, behind an access-logged function
that does not exist yet (out of this task's scope) -- so this module cannot
be used to accidentally peek at it during development.

CONTAINER-ONLY: every ``bits_per_parameter`` number this module can produce in
this environment is a harness-correctness sanity check, not a capacity result.
Reproducing the published 2.0 bits/param baseline needs ~1000 exposures per
fact and the M4 Mac mini target hardware (implementation/00_START_HERE.md
section 1, implementation/01_phases_and_gates.md P1). Label every number
accordingly when reporting it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from src.data.generate import write_dataset
from src.data.schema import ATTRIBUTES, Split, dataset_entropy_bits
from src.data.tokenizer import PAD, Tokenizer, build_tokenizer
from src.metrics.bits import (
    EntropyBoundViolation,
    answer_cross_entropy_bits,
    bits_per_parameter,
    bits_recovered_per_attribute,
    count_parameters,
)
from src.models.config import ModelConfig
from src.models.tiny_lm import TinyLM, causal_lm_shift
from src.train import train_loop

__all__ = [
    "EVAL_SPLITS",
    "IntegrationResult",
    "TrainingTextTooLongError",
    "default_tiny_config",
    "run_pipeline",
    "main",
]

# The only splits this module will evaluate on. Split.TRAIN has no
# question/answer records (it is declarative text) and Split.SEALED is
# deliberately excluded -- see the module docstring's "Scope note".
EVAL_SPLITS: Final[tuple[str, ...]] = (Split.PROBE, Split.UNSEEN_PERSON)

# Margin on the total-bits entropy-bound check, in bits. `bits_recovered_per_attribute`
# already enforces this per item at a tolerance of 1e-4 bits; summing up to
# len(ATTRIBUTES) already-checked attribute totals (each a float32 accumulation)
# can accumulate at most a few times that before this second, coarser check
# would false-trigger, so 1e-2 bits comfortably clears float32 rounding noise
# while remaining many orders of magnitude below any real bug (the nats/bits
# conversion error alone is a 44% multiplicative error -- see S1).
_TOTAL_ENTROPY_EPS_BITS: Final[float] = 1e-2


class TrainingTextTooLongError(ValueError):
    """Raised when a person's rendered training text exceeds ``config.block_size``.

    This pipeline never silently truncates training text: truncating a
    biography paragraph would drop whichever attribute sentences fall past the
    cut, silently lowering the true entropy of what the model was actually
    shown while ``schema.dataset_entropy_bits`` keeps assuming the full amount
    (implementation/03_edge_cases_and_scares.md, Training / Huge: "truncate
    explicitly and log, never wrap silently" -- for eval context this pipeline
    truncates loudly via that mechanism, but training data gets a hard error
    instead, since there is no caller-visible place to log a silent drop of
    ground truth). Raise and let the caller pick a larger ``block_size``.
    """


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Everything the P0 harness needs from one end-to-end pipeline run.

    Attributes:
        bits_recovered: bits recovered per attribute (keys are
            ``schema.ATTRIBUTES``), as returned by
            ``src.metrics.bits.bits_recovered_per_attribute``.
        total_bits_recovered: ``sum(bits_recovered.values())``.
        parameter_count: ``src.metrics.bits.count_parameters(model,
            include_embeddings=True)`` -- the full-trainable-parameters
            convention stated in ``src/models/config.py`` and
            ``src/metrics/bits.py``.
        bits_per_parameter: ``total_bits_recovered / parameter_count``.
            CONTAINER-ONLY: never a capacity result in this environment (see
            module docstring).
        n_people_scored: number of distinct people evaluated -- the
            denominator of ``entropy_bound_bits``.
        entropy_bound_bits: ``schema.dataset_entropy_bits(n_people_scored)``,
            the ceiling ``total_bits_recovered`` is asserted (live, in
            :func:`run_pipeline`) never to exceed.
        eval_split: which split (one of :data:`EVAL_SPLITS`) was scored.
        steps: optimizer steps actually run (0 means the untrained,
            randomly-initialised null model -- see the module docstring).
    """

    bits_recovered: dict[str, float]
    total_bits_recovered: float
    parameter_count: int
    bits_per_parameter: float
    n_people_scored: int
    entropy_bound_bits: float
    eval_split: str
    steps: int


def _read_jsonl(path: Path) -> list[dict]:
    """Read a ``write_dataset``-produced JSONL file into a list of dicts.

    Raises:
        json.JSONDecodeError: if a non-empty line is not valid JSON.
    """
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _tokenize_training_corpus(
    train_records: list[dict], tokenizer: Tokenizer, block_size: int
) -> torch.Tensor:
    """Tokenize every training paragraph and right-pad to ``block_size``.

    Right-padding (rather than left-padding) is what keeps this safe under
    this model's causal attention: a pad token at position ``t`` can only
    attend to (and only be attended *from* by later positions that are
    themselves never scored), so padding after the real content never
    perturbs the logits at any position that is actually used as a next-token
    training target for real content.

    Args:
        train_records: one dict per person, each with a ``"text"`` key (as
            written by ``write_dataset`` for the train split), and a
            ``"person_id"`` key used only for error messages.
        tokenizer: the closed-vocabulary tokenizer to encode with.
        block_size: target sequence length; every row of the returned tensor
            is exactly this long.

    Returns:
        Token ids, shape ``(len(train_records), block_size)``, dtype
        ``torch.long``.

    Raises:
        TrainingTextTooLongError: if any record's text encodes to more than
            ``block_size`` tokens.
        src.data.tokenizer.OutOfVocabularyError: propagated from ``encode``.
    """
    pad_id = tokenizer.encode(PAD)[0]
    rows: list[list[int]] = []
    for record in train_records:
        ids = tokenizer.encode(record["text"])
        if len(ids) > block_size:
            raise TrainingTextTooLongError(
                f"person {record['person_id']!r} rendered training text encodes to "
                f"{len(ids)} tokens, exceeding config.block_size={block_size}; "
                f"increase block_size rather than truncating training data"
            )
        rows.append(ids + [pad_id] * (block_size - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def _evaluate(
    model: TinyLM, tokenizer: Tokenizer, qa_records: list[dict], block_size: int
) -> tuple[dict[str, list[float]], int]:
    """Score every (person, attribute) QA record and group CE_bits by attribute.

    One forward pass per record (batch size 1): QA sequences are short
    (<=~20 tokens here) and vary in length per attribute, so batching would
    require padding logic that buys nothing at this scale and is easy to get
    subtly wrong (a pad token inside the *prompt* region, rather than after
    all real content, would change attention for every later position).

    Args:
        model: the model to evaluate. Not mutated; only ``forward`` is called,
            under ``torch.no_grad()``.
        tokenizer: the closed-vocabulary tokenizer to encode with.
        qa_records: dicts with ``"attribute"``, ``"question"``, ``"answer"``,
            ``"person_id"`` keys, as written by ``write_dataset`` for the
            probe/unseen_person/sealed splits.
        block_size: the model's maximum context length.

    Returns:
        ``(ce_by_attribute, n_people_scored)``: a dict mapping each attribute
        name in ``schema.ATTRIBUTES`` to a list of per-person CE_bits values
        (as returned by ``answer_cross_entropy_bits``), and the count of
        distinct ``person_id`` values seen across all records.

    Raises:
        ValueError: if a record's ``question`` + ``answer`` encodes to more
            tokens than ``block_size`` (propagated context, not silently
            truncated -- symmetric with :class:`TrainingTextTooLongError`).
        src.data.tokenizer.OutOfVocabularyError: propagated from ``encode_qa``.
    """
    model.eval()
    ce_by_attribute: dict[str, list[float]] = {attr: [] for attr in ATTRIBUTES}
    seen_people: set[str] = set()
    with torch.no_grad():
        for record in qa_records:
            attribute = record["attribute"]
            ids, answer_mask = tokenizer.encode_qa(record["question"], record["answer"])
            if len(ids) > block_size:
                raise ValueError(
                    f"probe question+answer for person={record['person_id']!r} "
                    f"attribute={attribute!r} encodes to {len(ids)} tokens, exceeding "
                    f"config.block_size={block_size}"
                )
            ids_t = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.tensor([answer_mask], dtype=torch.bool)
            logits = model(ids_t)
            shifted_logits, targets, shifted_mask = causal_lm_shift(logits, ids_t, mask_t)
            ce_bits = answer_cross_entropy_bits(shifted_logits, targets, shifted_mask)
            ce_by_attribute[attribute].append(ce_bits)
            seen_people.add(record["person_id"])
    return ce_by_attribute, len(seen_people)


def default_tiny_config(tokenizer: Tokenizer | None = None, *, block_size: int = 128) -> ModelConfig:
    """A small ``ModelConfig`` sized for this pipeline's real tokenizer vocabulary.

    Deliberately not one of ``src.models.config.PRESET_CONFIGS``: those presets
    fix ``vocab_size=256`` (a byte-level placeholder, per that module's
    docstring) which does not match this project's real, closed word-level
    tokenizer (``vocab_size == 2818``). Building a config here rather than
    editing ``PRESET_CONFIGS`` keeps ``src/models/config.py`` untouched, per
    this task's file scope.

    ``block_size=128`` is sized empirically: the longest observed rendered
    training paragraph over 500 sampled people was 76 tokens (worst case:
    longest template plus a 2-digit day, a long month name, and a 4-digit
    year); 128 leaves comfortable headroom without being large enough to slow
    the tiny-scale P0 tests down.

    Args:
        tokenizer: if given, its ``vocab_size`` is used; otherwise
            :func:`src.data.tokenizer.build_tokenizer` is called.
        block_size: passed straight through to ``ModelConfig``.

    Returns:
        A ``ModelConfig`` with ``n_layer=2, n_head=2, d_model=64,
        tie_embeddings=True`` and the tokenizer's real ``vocab_size``.
    """
    if tokenizer is None:
        tokenizer = build_tokenizer()
    return ModelConfig(
        n_layer=2,
        n_head=2,
        d_model=64,
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        tie_embeddings=True,
    )


def run_pipeline(
    n_people: int,
    seed: int,
    config: ModelConfig,
    steps: int,
    *,
    eval_split: str = Split.PROBE,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    num_threads: int = 1,
) -> IntegrationResult:
    """Run the full P0 pipeline once: generate -> tokenize -> train -> evaluate.

    Deterministic: identical ``(n_people, seed, config, steps, eval_split,
    batch_size, lr, weight_decay, num_threads)`` always produces the same
    dataset (via ``write_dataset(..., seed)``), the same model initialisation
    and training trajectory (via ``train_loop``'s own ``set_determinism(seed)``),
    and therefore the same result. Dataset generation happens in a fresh
    ``tempfile.TemporaryDirectory`` per call, so concurrent/repeated calls
    never collide and nothing is left on disk afterwards.

    ``steps=0`` is the **null-model control**: ``train_loop`` still builds the
    seeded, randomly-initialised model, optimizer, and batcher, then runs zero
    optimizer steps. This is the *same* code path as a trained run, not a
    separate one -- which is what makes it a legitimate C1 control
    (implementation/02_testing_philosophy.md section 2).

    Args:
        n_people: size of the main population (train + probe + unseen_person),
            per ``write_dataset``. Must satisfy that function's constraints
            (notably: not in ``{1, 2, 3, 4}``, since a 90/10 split of that few
            people would leave a split empty).
        seed: determinism seed, threaded through both dataset generation and
            training.
        config: model architecture. ``config.vocab_size`` must equal the real
            tokenizer's ``vocab_size`` (see :func:`default_tiny_config`).
            ``config.block_size`` must be large enough to hold every train
            record's tokenized text (see :class:`TrainingTextTooLongError`)
            and every eval record's tokenized question+answer.
        steps: total optimizer steps to run (0 = null model, see above).
        eval_split: which split to score bits-recovered on. Must be one of
            :data:`EVAL_SPLITS` (``Split.PROBE`` or ``Split.UNSEEN_PERSON`` --
            ``Split.SEALED`` is deliberately unreachable here, see the module
            docstring's "Scope note"; ``Split.TRAIN`` has no QA records).
        batch_size: training batch size, capped to the number of train
            examples if smaller (``ToyBatcher`` requires ``batch_size <=
            n_examples``).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (default 0.0, matching
            ``train_loop``'s own default and rationale).
        num_threads: pinned torch thread count, forwarded to
            ``train_loop``/``set_determinism`` (H3: needed for bit-identical
            re-runs).

    Returns:
        An :class:`IntegrationResult`.

    Raises:
        ValueError: if ``eval_split`` is not in :data:`EVAL_SPLITS`; if
            ``config.vocab_size`` does not match the tokenizer's; propagated
            from ``write_dataset`` for invalid ``(n_people, seed)``; propagated
            from ``bits_recovered_per_attribute`` if an attribute's evaluated
            list is empty (e.g. ``n_people`` too small for the requested
            split to contain anyone).
        TrainingTextTooLongError: if a train record's text does not fit
            ``config.block_size``.
        EntropyBoundViolation: if the measured total bits recovered exceeds
            ``schema.dataset_entropy_bits(n_people_scored)`` -- the metric or
            its input is broken; this is a live guard, not only a test
            assertion (implementation/04_golden_dataset.md section 4).
        src.data.tokenizer.OutOfVocabularyError: propagated from tokenizing
            any generated text (should be unreachable: the tokenizer's
            vocabulary is built to cover everything the generator can emit).
    """
    if eval_split not in EVAL_SPLITS:
        raise ValueError(
            f"eval_split must be one of {EVAL_SPLITS}, got {eval_split!r} "
            f"(Split.TRAIN has no question/answer records; Split.SEALED is "
            f"deliberately unreachable from this pipeline -- see module docstring)"
        )

    tokenizer = build_tokenizer()
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"config.vocab_size ({config.vocab_size}) != tokenizer.vocab_size "
            f"({tokenizer.vocab_size}); the model and tokenizer must share one vocabulary"
        )

    with tempfile.TemporaryDirectory(prefix="sparse_mamba_integration_") as tmp:
        out_dir = Path(tmp)
        manifest = write_dataset(out_dir, n_people, seed)

        train_records = _read_jsonl(out_dir / manifest["splits"][Split.TRAIN]["path"])
        data = _tokenize_training_corpus(train_records, tokenizer, config.block_size)

        train_result = train_loop(
            config=config,
            seed=seed,
            steps=steps,
            batch_size=min(batch_size, data.shape[0]),
            lr=lr,
            data=data,
            weight_decay=weight_decay,
            num_threads=num_threads,
        )
        model = train_result.model
        if model is None:  # pragma: no cover -- train_loop always sets this
            raise AssertionError("train_loop returned no model")

        eval_path = out_dir / manifest["splits"][eval_split]["path"]
        qa_records = _read_jsonl(eval_path)
        ce_by_attribute, n_scored = _evaluate(model, tokenizer, qa_records, config.block_size)

        bits_recovered = bits_recovered_per_attribute(ce_by_attribute)
        total_bits_recovered = float(sum(bits_recovered.values()))

        entropy_bound = dataset_entropy_bits(n_scored)
        if total_bits_recovered > entropy_bound + _TOTAL_ENTROPY_EPS_BITS:
            raise EntropyBoundViolation(
                f"total bits_recovered={total_bits_recovered} exceeds "
                f"dataset_entropy_bits({n_scored})={entropy_bound} for split "
                f"{eval_split!r}; the metric or its input is broken, this is never "
                f"a legitimate model result (implementation/04_golden_dataset.md section 4)"
            )

        parameter_count = count_parameters(model, include_embeddings=True)
        per_param = bits_per_parameter(total_bits_recovered, parameter_count)

    return IntegrationResult(
        bits_recovered=bits_recovered,
        total_bits_recovered=total_bits_recovered,
        parameter_count=parameter_count,
        bits_per_parameter=per_param,
        n_people_scored=n_scored,
        entropy_bound_bits=entropy_bound,
        eval_split=eval_split,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end P0 pipeline once and print the result as JSON. "
        "CONTAINER-ONLY: bits_per_parameter here is a harness sanity number, never a "
        "capacity result (implementation/00_START_HERE.md section 6)."
    )
    parser.add_argument("--n-people", type=int, required=True, help="dataset population size")
    parser.add_argument("--seed", type=int, required=True, help="determinism seed")
    parser.add_argument("--steps", type=int, required=True, help="optimizer steps (0 = null model)")
    parser.add_argument("--eval-split", type=str, default=Split.PROBE, choices=list(EVAL_SPLITS))
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=1, help="pinned torch thread count (H3)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 success, 1 on a live control firing)."""
    args = _build_arg_parser().parse_args(argv)
    tokenizer = build_tokenizer()
    config = ModelConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        tie_embeddings=True,
    )
    try:
        result = run_pipeline(
            n_people=args.n_people,
            seed=args.seed,
            config=config,
            steps=args.steps,
            eval_split=args.eval_split,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_threads=args.num_threads,
        )
    except EntropyBoundViolation as exc:
        print(f"ENTROPY BOUND VIOLATED: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "bits_recovered": result.bits_recovered,
                "total_bits_recovered": result.total_bits_recovered,
                "parameter_count": result.parameter_count,
                "bits_per_parameter_CONTAINER_ONLY": result.bits_per_parameter,
                "n_people_scored": result.n_people_scored,
                "entropy_bound_bits": result.entropy_bound_bits,
                "eval_split": result.eval_split,
                "steps": result.steps,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
