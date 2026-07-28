"""Lead-owned canary: RNG-state restore on resume is currently a NO-OP.

WHAT HAPPENED, INCLUDING MY OWN ERROR
-------------------------------------
During lead verification I removed the ``_restore_rng_state(...)`` call from
train_loop's resume path and re-ran the agent's resume tests. They PASSED. My
first conclusion was that the agent's tests were deficient. **That conclusion was
wrong**, and I am recording it rather than quietly deleting it.

I then wrote two tests here that assert RNG state round-trips through
save/load. They also passed with the break in place, because they call
``save_checkpoint``/``torch.set_rng_state`` directly and never exercise
train_loop's resume path. So my first fix was no better than what it criticised.

The actual explanation, proven empirically::

    checkpointed RNG state == post-construction RNG state  ->  True

``TinyLM`` is deliberately dropout-free, so a training step consumes **no**
randomness. ``train_loop`` also calls ``set_determinism(seed)`` itself on every
invocation. Together these mean the RNG state at checkpoint time is identical to
the state a fresh resumed process reaches after seeding and constructing the
model -- so restoring it changes nothing.

**The break is therefore undetectable by construction, not because any test is
weak.** No test can distinguish a no-op from its absence. The agent's resume
tests were correct; the property is currently vacuous.

WHAT THIS FILE IS FOR
---------------------
The restore code is correct and must stay, because the property stops being
vacuous the instant anything consumes randomness: dropout, sampling, stochastic
augmentation, or a data pipeline with random ordering. At that moment resume
would silently diverge from an uninterrupted run -- the exact failure mode that
invalidates an A/B comparison between architectures without ever raising an
error, in long runs that are expected to be interrupted.

So the load-bearing test here is the **canary**,
``test_model_is_currently_rng_free_documenting_why_the_gap_existed``. It fails
loudly the moment the precondition breaks, converting a latent silent trap into
an immediate, actionable failure. The two round-trip tests are kept because they
pin the save/load contract itself, and the train_loop test is kept because it
becomes meaningful the moment the canary fires -- but neither can fail today, and
that is documented rather than disguised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.config import PRESET_CONFIGS  # noqa: E402
from src.models.tiny_lm import TinyLM  # noqa: E402
from src.train import (  # noqa: E402
    ToyBatcher,
    load_checkpoint,
    make_toy_dataset,
    save_checkpoint,
    set_determinism,
    train_loop,
)


def _fresh_run(tmp_path: Path) -> tuple[TinyLM, torch.optim.Optimizer, ToyBatcher, Path]:
    config = PRESET_CONFIGS["tiny"]
    set_determinism(seed=0, num_threads=1)
    model = TinyLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = make_toy_dataset(n_examples=8, config=config, seed=0)
    batcher = ToyBatcher(data, batch_size=4, seed=0)
    return model, optimizer, batcher, tmp_path / "ckpt.pt"


def test_rng_state_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """The saved RNG state must be byte-identical to the live state at save time.

    This is the invariant the agent's weight-equality resume tests cannot see,
    because a dropout-free model consumes no RNG.
    """
    model, optimizer, batcher, ckpt = _fresh_run(tmp_path)

    # Consume some randomness so the state is non-trivial and position-dependent.
    torch.randn(64)
    expected_torch_state = torch.get_rng_state().clone()

    save_checkpoint(
        path=ckpt,
        step=1,
        model=model,
        optimizer=optimizer,
        batcher=batcher,
        seed=0,
        config=PRESET_CONFIGS["tiny"],
        loss_history=[1.0],
    )
    payload = load_checkpoint(ckpt)

    assert torch.equal(payload["rng_state"]["torch"], expected_torch_state), (
        "checkpoint did not capture the live torch RNG state"
    )


def test_restoring_checkpoint_rng_reproduces_the_same_draw(tmp_path: Path) -> None:
    """After restoring, the next random draw must match the uninterrupted run.

    Direct behavioural check: the whole point of restoring RNG state is that the
    stream continues where it left off. Perturbing the global RNG between save
    and restore simulates an interrupted-and-resumed process.
    """
    model, optimizer, batcher, ckpt = _fresh_run(tmp_path)
    torch.randn(64)

    save_checkpoint(
        path=ckpt,
        step=1,
        model=model,
        optimizer=optimizer,
        batcher=batcher,
        seed=0,
        config=PRESET_CONFIGS["tiny"],
        loss_history=[1.0],
    )

    # What an uninterrupted run would have drawn next.
    uninterrupted_next = torch.randn(8).clone()

    # Simulate a separate process with a different RNG position.
    torch.manual_seed(999_999)
    torch.randn(1234)
    assert not torch.equal(torch.randn(8), uninterrupted_next), (
        "RNG perturbation did not take effect; this test would be vacuous"
    )

    # Restore exactly as train_loop's resume path does.
    payload = load_checkpoint(ckpt)
    torch.set_rng_state(payload["rng_state"]["torch"])
    resumed_next = torch.randn(8)

    assert torch.equal(resumed_next, uninterrupted_next), (
        "resumed RNG stream diverged from the uninterrupted run; "
        "resume is not bit-reproducible once any RNG-consuming op exists"
    )


def test_train_loop_resume_leaves_rng_where_uninterrupted_run_would(tmp_path: Path) -> None:
    """train_loop resume must leave RNG where an uninterrupted run would.

    CURRENTLY VACUOUS -- and deliberately kept. See the module docstring: with a
    dropout-free model the restore is a proven no-op, so this test cannot fail
    today even with the restore deleted. It is retained because it becomes the
    real regression test the moment the canary below fires.

    Do not read a pass here as evidence that RNG restore works. Read it as
    evidence that nothing currently depends on it.
    """
    config = PRESET_CONFIGS["tiny"]
    data = make_toy_dataset(n_examples=8, config=config, seed=0)
    common = dict(config=config, seed=0, batch_size=4, lr=1e-3, data=data, num_threads=1)

    # Uninterrupted reference run.
    train_loop(steps=6, out_dir=tmp_path / "ref", **common)  # type: ignore[arg-type]
    reference_rng = torch.get_rng_state().clone()

    # Checkpoint at step 3, then resume to step 6 in a "fresh process" whose
    # RNG has been perturbed, exactly as a real restart would be.
    train_loop(steps=3, out_dir=tmp_path / "part", checkpoint_every=3, **common)  # type: ignore[arg-type]
    ckpt = tmp_path / "part" / "checkpoint_step3.pt"
    if not ckpt.exists():
        candidates = sorted((tmp_path / "part").glob("*.pt"))
        assert candidates, f"no checkpoint written into {tmp_path / 'part'}"
        ckpt = candidates[-1]

    torch.manual_seed(1_234_567)
    torch.randn(999)

    train_loop(steps=6, out_dir=tmp_path / "res", resume_path=ckpt, **common)  # type: ignore[arg-type]
    resumed_rng = torch.get_rng_state()

    assert torch.equal(resumed_rng, reference_rng), (
        "global RNG state after resume differs from the uninterrupted run: "
        "train_loop is not restoring RNG state, so any future RNG-consuming op "
        "(dropout, sampling, augmentation) would make resume silently diverge"
    )


def test_model_is_currently_rng_free_documenting_why_the_gap_existed() -> None:
    """Documents the precondition that made the gap invisible.

    If this test ever FAILS, TinyLM has gained an RNG-consuming op (dropout,
    sampling, stochastic augmentation). That is fine -- but it means the resume
    path is now load-bearing for reproducibility, and the weight-equality resume
    tests have become genuinely sensitive to RNG restore. Treat a failure here
    as a prompt to re-check that resume is still bit-identical, not as a bug.
    """
    config = PRESET_CONFIGS["tiny"]
    set_determinism(seed=0, num_threads=1)
    model = TinyLM(config)
    model.train()
    batch = torch.randint(0, config.vocab_size, (2, 16))

    before = torch.get_rng_state().clone()
    model(batch)
    after = torch.get_rng_state()

    if not torch.equal(before, after):
        pytest.fail(
            "TinyLM now consumes RNG in training mode. The resume tests are now "
            "sensitive to RNG restore -- re-verify bit-identical resume and "
            "update this test's docstring."
        )
