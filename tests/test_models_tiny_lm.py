"""Tests for the baseline TinyLM model and training loop (src/models/, src/train.py).

Read implementation/02_testing_philosophy.md before extending this file: a test
that only passes when the code works is worth very little. Every test targeting
a specific requirement is designed to FAIL if the bug it targets is present --
see the "deliberate breakage" verification log in the task report for evidence
that each one actually does.

Test-to-requirement map (implementation/06_agent_briefs/B3_baseline_model.md):

* R1 (analytic == actual param count): test_analytic_param_count_matches_actual_presets,
  test_analytic_param_count_matches_actual_untied (>= 3 configs total).
* R3 (determinism): test_determinism_same_seed_identical_weights (+ a negative
  control, test_determinism_different_seed_differs).
* R4/H4 (checkpoint+resume): test_resume_produces_bit_identical_final_weights,
  test_checkpoint_roundtrip_preserves_every_field, test_resume_rejects_mismatched_config,
  test_fresh_run_into_occupied_out_dir_fails_fast.
* R5 (logits hook): test_causal_lm_shift_shapes_and_dtypes,
  test_causal_lm_shift_consumable_by_answer_cross_entropy_bits.
* R6 (NaN/Inf abort): test_nan_loss_aborts_with_step_number, test_inf_loss_also_aborts.
* "the most important" overfit test: test_overfit_drives_loss_near_zero_on_20_examples.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch

import src.train as train_module
from src.metrics.bits import answer_cross_entropy_bits, count_parameters
from src.models.config import PRESET_CONFIGS, ModelConfig, analytic_param_count
from src.models.tiny_lm import TinyLM, causal_lm_shift, truncate_ids
from src.train import (
    ToyBatcher,
    TrainingDivergedError,
    load_checkpoint,
    make_toy_dataset,
    save_checkpoint,
    train_loop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        n_layer=2, n_head=2, d_model=32, vocab_size=32, block_size=16, tie_embeddings=True
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _assert_state_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"parameter {key!r} differs"


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _make_permutation_dataset(n_examples: int, vocab_size: int, block_size: int, seed: int) -> torch.Tensor:
    """A toy dataset where every example is a permutation of the SAME fixed multiset.

    Deliberately chosen so that predicting the next token requires knowing
    *which slot in the sequence* the model is at, not merely which tokens
    have appeared so far. Verified empirically (see task report) that a
    plain i.i.d.-random toy dataset does **not** exercise this: causal
    self-attention's mask alone carries enough implicit positional signal
    that dropping `TinyLM.wpe` barely changes convergence on i.i.d. data,
    which would make the overfit test blind to the "drop positional
    embeddings" break the acceptance table requires it to catch
    (implementation/06_agent_briefs/B3_baseline_model.md). This
    permutation-of-a-fixed-base construction produces a measurable,
    deterministic gap instead (see test docstring below for the numbers).
    """
    generator = torch.Generator().manual_seed(seed)
    base = torch.randint(0, vocab_size, (block_size,), generator=generator)
    rows = [base[torch.randperm(block_size, generator=generator)] for _ in range(n_examples)]
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# 1. ModelConfig validation (edge-case ladder: malformed / boundary)
# ---------------------------------------------------------------------------


def test_config_rejects_non_positive_dims():
    for field, bad in (("n_layer", 0), ("n_head", 0), ("d_model", 0), ("vocab_size", 0), ("n_layer", -1)):
        with pytest.raises(ValueError):
            _tiny_config(**{field: bad})


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="evenly divisible"):
        _tiny_config(d_model=33, n_head=2)


def test_config_rejects_block_size_below_2():
    with pytest.raises(ValueError, match="block_size"):
        _tiny_config(block_size=1)


def test_config_rejects_non_int_fields():
    with pytest.raises(TypeError):
        _tiny_config(n_layer=2.0)
    with pytest.raises(TypeError):
        _tiny_config(d_model=True)  # bool is not accepted even though bool is an int subclass


def test_config_rejects_non_bool_tie_embeddings():
    with pytest.raises(TypeError):
        _tiny_config(tie_embeddings=1)


# ---------------------------------------------------------------------------
# 2. Analytic == actual parameter count (R1) -- required test 1, >= 3 configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", ["1m", "10m", "100m"])
def test_analytic_param_count_matches_actual_presets(preset_name):
    """The convention (stated in src/models/config.py's module docstring and
    src/metrics/bits.py's count_parameters docstring): all trainable
    parameters, tied parameters counted once. Two independently-derived
    counts -- a closed-form formula here, and iterating the instantiated
    model's named_parameters() in src.metrics.bits -- must agree exactly.
    """
    config = PRESET_CONFIGS[preset_name]
    model = TinyLM(config)
    analytic = analytic_param_count(config)
    actual = count_parameters(model, include_embeddings=True)
    assert analytic == actual, f"{preset_name}: analytic={analytic} actual={actual}"
    assert analytic > 0


def test_analytic_param_count_matches_actual_untied():
    """A 4th, non-preset config exercising tie_embeddings=False (the analytic
    formula's `if not tie_embeddings: total += vocab*d` branch)."""
    config = _tiny_config(tie_embeddings=False, vocab_size=40, d_model=48, n_head=4)
    model = TinyLM(config)
    assert analytic_param_count(config) == count_parameters(model, include_embeddings=True)


def test_tied_vs_untied_differ_by_exactly_vocab_times_d():
    tied = _tiny_config(tie_embeddings=True, vocab_size=40, d_model=48, n_head=4)
    untied = _tiny_config(tie_embeddings=False, vocab_size=40, d_model=48, n_head=4)
    assert analytic_param_count(untied) - analytic_param_count(tied) == 40 * 48


def test_embedding_exclusion_convention_matches_structural_detection():
    """States the include_embeddings=False convention explicitly (S3): under
    this architecture both `wte` and `wpe` are `nn.Embedding` modules, so
    excluding embeddings strips exactly `vocab_size*d + block_size*d`
    (never `lm_head`, even when untied -- it is an `nn.Linear`, not an
    `nn.Embedding`). If this test ever disagreed with
    src.metrics.bits.count_parameters's structural detection, that would be
    exactly the S3 failure mode (a silent denominator mismatch) and must be
    reported loudly, not papered over.
    """
    config = PRESET_CONFIGS["1m"]
    model = TinyLM(config)
    d = config.d_model
    expected_embedding_params = config.vocab_size * d + config.block_size * d

    with_embeddings = count_parameters(model, include_embeddings=True)
    without_embeddings = count_parameters(model, include_embeddings=False)
    assert with_embeddings - without_embeddings == expected_embedding_params


def test_preset_param_counts_are_in_the_right_ballpark():
    """Sanity check on R2 ('configs for ~1M, ~10M, ~100M') -- within 3x of
    the named target, which is what 'ballpark' means here; the exact
    numbers are asserted precisely by the analytic-vs-actual tests above."""
    targets = {"1m": 1_000_000, "10m": 10_000_000, "100m": 100_000_000}
    for name, target in targets.items():
        actual = analytic_param_count(PRESET_CONFIGS[name])
        assert target / 3 < actual < target * 3, f"{name}: {actual} not within 3x of {target}"


# ---------------------------------------------------------------------------
# 3. TinyLM.forward edge cases (edge-case ladder)
# ---------------------------------------------------------------------------


def test_forward_output_shape_and_dtype():
    config = _tiny_config()
    model = TinyLM(config)
    idx = torch.randint(0, config.vocab_size, (3, config.block_size))
    logits = model(idx)
    assert logits.shape == (3, config.block_size, config.vocab_size)
    assert logits.dtype == torch.float32


def test_forward_rejects_float_dtype():
    config = _tiny_config()
    model = TinyLM(config)
    idx = torch.zeros(2, config.block_size, dtype=torch.float32)
    with pytest.raises(TypeError):
        model(idx)


def test_forward_rejects_wrong_ndim():
    config = _tiny_config()
    model = TinyLM(config)
    with pytest.raises(ValueError, match="2D"):
        model(torch.zeros(config.block_size, dtype=torch.long))


def test_forward_rejects_empty_batch():
    config = _tiny_config()
    model = TinyLM(config)
    with pytest.raises(ValueError, match="batch size 0"):
        model(torch.zeros(0, config.block_size, dtype=torch.long))


def test_forward_rejects_empty_sequence():
    config = _tiny_config()
    model = TinyLM(config)
    with pytest.raises(ValueError, match="seq_len 0"):
        model(torch.zeros(2, 0, dtype=torch.long))


def test_forward_rejects_seq_len_exceeding_block_size():
    config = _tiny_config()
    model = TinyLM(config)
    idx = torch.zeros(2, config.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_forward_rejects_out_of_range_token_id():
    config = _tiny_config()
    model = TinyLM(config)
    idx = torch.zeros(2, config.block_size, dtype=torch.long)
    idx[0, 0] = config.vocab_size  # one past the end
    with pytest.raises(ValueError, match="outside"):
        model(idx)
    idx[0, 0] = -1
    with pytest.raises(ValueError, match="outside"):
        model(idx)


def test_truncate_ids_no_op_when_within_block_size():
    ids = torch.arange(10)
    assert torch.equal(truncate_ids(ids, 10), ids)
    assert torch.equal(truncate_ids(ids, 20), ids)


def test_truncate_ids_truncates_and_warns():
    ids = torch.arange(10)
    with pytest.warns(UserWarning, match="exceeds block_size"):
        truncated = truncate_ids(ids, 4)
    assert torch.equal(truncated, torch.tensor([6, 7, 8, 9]))


def test_truncate_ids_rejects_non_positive_block_size():
    with pytest.raises(ValueError):
        truncate_ids(torch.arange(5), 0)


# ---------------------------------------------------------------------------
# 4. Logits hook contract (R5) -- required test 5
# ---------------------------------------------------------------------------


def test_causal_lm_shift_shapes_and_dtypes():
    config = _tiny_config()
    model = TinyLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.block_size))
    answer_mask = torch.zeros(3, config.block_size, dtype=torch.bool)
    answer_mask[:, -2:] = True

    logits = model(input_ids)
    shifted_logits, targets, shifted_mask = causal_lm_shift(logits, input_ids, answer_mask)

    assert shifted_logits.shape == (3, config.block_size - 1, config.vocab_size)
    assert targets.shape == (3, config.block_size - 1)
    assert shifted_mask.shape == (3, config.block_size - 1)
    assert shifted_logits.dtype == torch.float32
    assert targets.dtype == input_ids.dtype
    assert shifted_mask.dtype == torch.bool
    assert torch.equal(targets, input_ids[:, 1:])
    assert torch.equal(shifted_mask, answer_mask[:, 1:])


def test_causal_lm_shift_consumable_by_answer_cross_entropy_bits():
    """The strongest form of 'consumable by src/metrics/bits.py' -- actually
    calls answer_cross_entropy_bits with this function's output and checks
    it runs and returns a sane value, rather than only checking shapes."""
    config = _tiny_config()
    model = TinyLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))
    answer_mask = torch.zeros(2, config.block_size, dtype=torch.bool)
    answer_mask[:, -3:] = True

    with torch.no_grad():
        logits = model(input_ids)
    shifted_logits, targets, shifted_mask = causal_lm_shift(logits, input_ids, answer_mask)
    bits = answer_cross_entropy_bits(shifted_logits, targets, shifted_mask)
    assert math.isfinite(bits)
    assert bits >= 0.0


def test_causal_lm_shift_rejects_seq_len_below_2():
    logits = torch.zeros(2, 1, 10)
    ids = torch.zeros(2, 1, dtype=torch.long)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="seq_len must be >= 2"):
        causal_lm_shift(logits, ids, mask)


def test_causal_lm_shift_rejects_mask_shape_mismatch():
    logits = torch.zeros(2, 5, 10)
    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.zeros(2, 4, dtype=torch.bool)  # wrong length
    with pytest.raises(ValueError, match="answer_mask"):
        causal_lm_shift(logits, ids, mask)


def test_causal_lm_shift_rejects_logits_ids_shape_mismatch():
    logits = torch.zeros(2, 5, 10)
    ids = torch.zeros(3, 5, dtype=torch.long)  # wrong batch dim
    mask = torch.zeros(3, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="incompatible"):
        causal_lm_shift(logits, ids, mask)


# ---------------------------------------------------------------------------
# 5. ToyBatcher: determinism / resumability building block
# ---------------------------------------------------------------------------


def test_toybatcher_rejects_bad_batch_size():
    data = torch.randint(0, 10, (5, 4))
    with pytest.raises(ValueError):
        ToyBatcher(data, batch_size=0, seed=0)
    with pytest.raises(ValueError):
        ToyBatcher(data, batch_size=6, seed=0)  # exceeds n_examples


def test_toybatcher_rejects_empty_data():
    with pytest.raises(ValueError):
        ToyBatcher(torch.zeros(0, 4, dtype=torch.long), batch_size=1, seed=0)


def test_toybatcher_reshuffles_into_new_epoch():
    data = torch.arange(20).reshape(5, 4)
    batcher = ToyBatcher(data, batch_size=2, seed=0)
    seen_rows = set()
    for _ in range(20):  # far more than one epoch (5 // 2 = 2 batches/epoch)
        batch = batcher.next_batch()
        for row in batch:
            seen_rows.add(int(row[0]))
    assert seen_rows == {0, 4, 8, 12, 16}  # every row's first element, all 5 rows visited


def test_toybatcher_state_dict_roundtrip_reproduces_identical_future_batches():
    data = torch.randint(0, 50, (10, 6))
    a = ToyBatcher(data, batch_size=3, seed=0)
    b = ToyBatcher(data, batch_size=3, seed=0)

    # advance `a` partway, then clone its exact state into `b`
    a.next_batch()
    a.next_batch()
    b.next_batch()
    b.next_batch()
    assert torch.equal(a.state_dict()["order"], b.state_dict()["order"])

    # advance `a` further; snapshot b's state now, advance b too, then rewind
    # b via load_state_dict and confirm it reproduces the same future batches
    snapshot = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.state_dict().items()}
    future_a = [a.next_batch().clone() for _ in range(5)]
    for _ in range(5):
        b.next_batch()  # diverge b
    b.load_state_dict(snapshot)
    future_b = [b.next_batch().clone() for _ in range(5)]
    for x, y in zip(future_a, future_b):
        assert torch.equal(x, y)


def test_make_toy_dataset_rejects_non_positive_n_examples():
    config = _tiny_config()
    with pytest.raises(ValueError):
        make_toy_dataset(0, config, seed=0)


def test_make_toy_dataset_is_deterministic():
    config = _tiny_config()
    a = make_toy_dataset(10, config, seed=5)
    b = make_toy_dataset(10, config, seed=5)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 6. Determinism (R3) -- required test 2
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_weights():
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=123)
    kwargs = dict(config=config, seed=42, steps=5, batch_size=4, lr=1e-3, data=data, num_threads=1)
    r1 = train_loop(**kwargs)
    r2 = train_loop(**kwargs)
    _assert_state_dicts_equal(r1.model.state_dict(), r2.model.state_dict())
    assert _state_dict_sha256(r1.model.state_dict()) == _state_dict_sha256(r2.model.state_dict())
    assert r1.loss_history == r2.loss_history


def test_determinism_different_seed_differs():
    """Negative control for the test above (implementation/00_START_HERE.md
    N1 mutation check): if the seed were silently ignored, the identical-
    weights assertion above would be vacuously true. Confirms it is not."""
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=123)
    r1 = train_loop(config=config, seed=1, steps=5, batch_size=4, lr=1e-3, data=data, num_threads=1)
    r2 = train_loop(config=config, seed=2, steps=5, batch_size=4, lr=1e-3, data=data, num_threads=1)
    assert _state_dict_sha256(r1.model.state_dict()) != _state_dict_sha256(r2.model.state_dict())


# ---------------------------------------------------------------------------
# 7. Checkpoint + resume correctness (R4, H4) -- required test 3
# ---------------------------------------------------------------------------


def test_resume_produces_bit_identical_final_weights(tmp_path: Path):
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=7)
    common = dict(config=config, seed=99, batch_size=4, lr=1e-3, data=data, num_threads=1)

    straight = train_loop(steps=10, **common)

    out_dir = tmp_path / "resume_run"
    train_loop(steps=5, out_dir=out_dir, **common)
    resumed = train_loop(steps=10, out_dir=out_dir, resume_path=out_dir / "final.pt", **common)

    _assert_state_dicts_equal(straight.model.state_dict(), resumed.model.state_dict())
    assert _state_dict_sha256(straight.model.state_dict()) == _state_dict_sha256(resumed.model.state_dict())
    assert straight.loss_history == resumed.loss_history
    assert resumed.final_step == 10


def test_resume_with_periodic_checkpoints_also_matches(tmp_path: Path):
    """Same property, exercised through a mid-training (`step_5.pt`) checkpoint
    rather than only the final one, and resuming to a *later* target than the
    interrupted run's original target."""
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=3)
    common = dict(config=config, seed=11, batch_size=4, lr=1e-3, data=data, num_threads=1)

    straight = train_loop(steps=12, **common)

    out_dir = tmp_path / "periodic"
    train_loop(steps=6, out_dir=out_dir, checkpoint_every=3, **common)
    resumed = train_loop(steps=12, out_dir=out_dir, resume_path=out_dir / "step_6.pt", **common)

    _assert_state_dicts_equal(straight.model.state_dict(), resumed.model.state_dict())


def test_checkpoint_roundtrip_preserves_every_field(tmp_path: Path):
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)
    result = train_loop(config=config, seed=1, steps=3, batch_size=4, lr=1e-3, data=data, num_threads=1)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        step=3,
        model=result.model,
        optimizer=result.optimizer,
        batcher=result.batcher,
        seed=1,
        config=config,
        loss_history=result.loss_history,
    )
    loaded = load_checkpoint(path)

    assert loaded["step"] == 3
    assert loaded["seed"] == 1
    assert loaded["loss_history"] == result.loss_history
    _assert_state_dicts_equal(loaded["model_state"], result.model.state_dict())
    assert loaded["optimizer_state"]["state"].keys() == result.optimizer.state_dict()["state"].keys()
    assert torch.equal(loaded["data_state"]["order"], result.batcher.state_dict()["order"])
    assert loaded["data_state"]["pos"] == result.batcher.state_dict()["pos"]
    for rng_key in ("python", "numpy", "torch", "torch_cuda"):
        assert rng_key in loaded["rng_state"]


def test_load_checkpoint_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_checkpoint(Path("/nonexistent/checkpoint/path/does/not/exist.pt"))


def test_resume_rejects_mismatched_config(tmp_path: Path):
    config_a = _tiny_config(d_model=32)
    config_b = _tiny_config(d_model=64, n_head=4)
    data = make_toy_dataset(20, config_a, seed=1)
    out_dir = tmp_path / "mismatch"
    train_loop(config=config_a, seed=1, steps=2, batch_size=4, lr=1e-3, data=data, out_dir=out_dir, num_threads=1)

    data_b = make_toy_dataset(20, config_b, seed=1)
    with pytest.raises(ValueError, match="cannot resume"):
        train_loop(
            config=config_b,
            seed=1,
            steps=4,
            batch_size=4,
            lr=1e-3,
            data=data_b,
            out_dir=out_dir,
            resume_path=out_dir / "final.pt",
            num_threads=1,
        )


def test_fresh_run_into_occupied_out_dir_fails_fast(tmp_path: Path):
    """implementation/03_edge_cases_and_scares.md, Training / Concurrent: two
    runs into the same output dir must fail fast, not interleave."""
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)
    out_dir = tmp_path / "occupied"
    train_loop(config=config, seed=1, steps=2, batch_size=4, lr=1e-3, data=data, out_dir=out_dir, num_threads=1)

    with pytest.raises(FileExistsError, match="already contains checkpoint"):
        train_loop(config=config, seed=1, steps=2, batch_size=4, lr=1e-3, data=data, out_dir=out_dir, num_threads=1)


def test_zero_steps_is_a_valid_boundary(tmp_path: Path):
    """Training / Boundary: 0 steps must not crash."""
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)
    result = train_loop(config=config, seed=1, steps=0, batch_size=4, lr=1e-3, data=data, num_threads=1)
    assert result.final_step == 0
    assert result.loss_history == []


# ---------------------------------------------------------------------------
# 8. NaN/Inf loss aborts immediately (R6)
# ---------------------------------------------------------------------------


def test_nan_loss_aborts_with_step_number(monkeypatch: pytest.MonkeyPatch):
    """Directly and deterministically triggers the guard by monkeypatching
    the loss computation to return NaN on a known step, rather than relying
    on a learning rate large enough to numerically diverge (which would make
    *which* step it happens on non-deterministic and the test flaky)."""
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)
    real_loss_fn = train_module._compute_lm_loss
    call_count = {"n": 0}

    def fake_loss(model, batch):
        call_count["n"] += 1
        if call_count["n"] == 3:
            return torch.tensor(float("nan"))
        return real_loss_fn(model, batch)

    monkeypatch.setattr(train_module, "_compute_lm_loss", fake_loss)

    with pytest.raises(TrainingDivergedError, match=r"NaN.*step 3"):
        train_loop(config=config, seed=1, steps=10, batch_size=4, lr=1e-3, data=data, num_threads=1)
    assert call_count["n"] == 3, "must abort immediately -- no further steps after the diverged one"


def test_inf_loss_also_aborts(monkeypatch: pytest.MonkeyPatch):
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)

    def fake_loss(model, batch):
        return torch.tensor(float("inf"))

    monkeypatch.setattr(train_module, "_compute_lm_loss", fake_loss)

    with pytest.raises(TrainingDivergedError, match=r"Inf.*step 1"):
        train_loop(config=config, seed=1, steps=10, batch_size=4, lr=1e-3, data=data, num_threads=1)


def test_no_checkpoint_written_for_diverged_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _tiny_config()
    data = make_toy_dataset(20, config, seed=1)
    out_dir = tmp_path / "diverged"

    def fake_loss(model, batch):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(train_module, "_compute_lm_loss", fake_loss)

    with pytest.raises(TrainingDivergedError):
        train_loop(
            config=config, seed=1, steps=10, batch_size=4, lr=1e-3, data=data,
            out_dir=out_dir, checkpoint_every=1, num_threads=1,
        )
    assert not list(out_dir.glob("*.pt"))


# ---------------------------------------------------------------------------
# 9. Overfit test (brief's required test 4 -- "the most important")
# ---------------------------------------------------------------------------

_OVERFIT_CONFIG = ModelConfig(n_layer=2, n_head=2, d_model=48, vocab_size=12, block_size=64, tie_embeddings=True)
_OVERFIT_STEPS = 400
_OVERFIT_LOSS_THRESHOLD = 0.025


def test_overfit_drives_loss_near_zero_on_20_examples(capsys: pytest.CaptureFixture[str]):
    """The single most important test in this file
    (implementation/06_agent_briefs/B3_baseline_model.md): the one thing
    that catches "the training loop silently does nothing." Must genuinely
    fail if optimizer.step() is removed -- verified for real, see the task
    report's break/revert log, not merely asserted here.

    Uses a permutation-of-a-fixed-base dataset (see
    _make_permutation_dataset), not i.i.d. random tokens: i.i.d. random data
    was tried first and found NOT to depend on positional embeddings (see
    task report) because causal self-attention's mask alone carries enough
    implicit ordering signal for that easier task -- which would make this
    test blind to the acceptance table's "drop positional embeddings ->
    overfit test (loss plateaus higher)" requirement. With this harder,
    genuinely position-dependent task, empirically (fixed seed, exactly
    reproducible under a fixed thread count -- see set_determinism's H3
    note):

    * working implementation converges to ~0.018 nats
    * optimizer.step() removed: flat at chance, ~2.51 nats (ln(12) = 2.4849)
    * positional embeddings dropped: plateaus at ~0.030 nats

    0.025 sits strictly between the working value and both broken values.
    """
    config = _OVERFIT_CONFIG
    data = _make_permutation_dataset(20, config.vocab_size, config.block_size, seed=0)
    chance_level = math.log(config.vocab_size)

    result = train_loop(
        config=config, seed=0, steps=_OVERFIT_STEPS, batch_size=20, lr=5e-3,
        data=data, num_threads=4, weight_decay=0.0,
    )

    curve_sample = [round(result.loss_history[i], 4) for i in range(0, _OVERFIT_STEPS, 40)]
    print(f"OVERFIT LOSS CURVE (chance level = {chance_level:.4f}, every 40 steps):")
    print(curve_sample)
    print(f"final loss (step {_OVERFIT_STEPS}): {result.loss_history[-1]:.6f}")

    assert result.loss_history[0] > chance_level * 0.8, "sanity: should start near chance level"
    assert result.loss_history[-1] < _OVERFIT_LOSS_THRESHOLD, (
        f"final loss {result.loss_history[-1]} did not drop below {_OVERFIT_LOSS_THRESHOLD}; "
        f"chance level is {chance_level:.4f}"
    )


# ---------------------------------------------------------------------------
# 10. CLI (train.py's `main`) -- exercises the actual entry point, not just
# the library functions it calls, so a broken argparse wiring or a broken
# exit-code path would be caught here rather than only by manual `python -m
# src.train ...` runs (which do not run in CI).
# ---------------------------------------------------------------------------


def test_cli_main_fresh_run_writes_checkpoint_and_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out_dir = tmp_path / "cli_fresh"
    exit_code = train_module.main(
        ["--seed", "0", "--config", "tiny", "--steps", "5", "--batch-size", "4",
         "--out-dir", str(out_dir), "--num-threads", "1"]
    )
    assert exit_code == 0
    assert (out_dir / "final.pt").exists()
    out = capsys.readouterr().out
    assert "final step 5" in out


def test_cli_main_resume_continues_to_new_target(tmp_path: Path):
    out_dir = tmp_path / "cli_resume"
    train_module.main(
        ["--seed", "1", "--config", "tiny", "--steps", "5", "--batch-size", "4",
         "--out-dir", str(out_dir), "--num-threads", "1"]
    )
    exit_code = train_module.main(
        ["--seed", "1", "--config", "tiny", "--steps", "9", "--batch-size", "4",
         "--out-dir", str(out_dir), "--num-threads", "1", "--resume", str(out_dir / "final.pt")]
    )
    assert exit_code == 0
    checkpoint = load_checkpoint(out_dir / "final.pt")
    assert checkpoint["step"] == 9


def test_cli_main_returns_one_on_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def fake_loss(model, batch):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(train_module, "_compute_lm_loss", fake_loss)

    exit_code = train_module.main(
        ["--seed", "0", "--config", "tiny", "--steps", "5", "--batch-size", "4",
         "--out-dir", str(tmp_path / "cli_diverge"), "--num-threads", "1"]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "TRAINING DIVERGED" in err
    assert "step 1" in err


def test_cli_main_second_fresh_run_into_occupied_dir_raises(tmp_path: Path):
    """The CLI does not swallow the concurrent-run guard -- it is not caught
    in `main`, so it propagates as an uncaught exception (nonzero process
    exit via the normal Python traceback path), never a silent overwrite."""
    out_dir = tmp_path / "cli_occupied"
    train_module.main(
        ["--seed", "0", "--config", "tiny", "--steps", "2", "--batch-size", "4",
         "--out-dir", str(out_dir), "--num-threads", "1"]
    )
    with pytest.raises(FileExistsError):
        train_module.main(
            ["--seed", "0", "--config", "tiny", "--steps", "2", "--batch-size", "4",
             "--out-dir", str(out_dir), "--num-threads", "1"]
        )
