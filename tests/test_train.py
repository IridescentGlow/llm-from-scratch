import numpy as np
import pytest
import torch

from llm_from_scratch.data import TokenDataset
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.tokenizer import BPETokenizer
from llm_from_scratch.train import (
    TrainConfig,
    estimate_loss,
    get_lr,
    load_checkpoint_for_resume,
    load_config,
    train_model,
)
from llm_from_scratch.train.loop import get_dataloader


def _tiny_model_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=20,
        context_length=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _tiny_train_config(**overrides) -> TrainConfig:
    defaults = dict(
        batch_size=4,
        learning_rate=1e-2,
        warmup_steps=2,
        max_steps=5,
        grad_clip=1.0,
        eval_every=5,
        checkpoint_dir="",  # set per-test via tmp_path
        checkpoint_every=5,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _tiny_dataset(vocab_size: int, context_length: int, length: int = 200) -> TokenDataset:
    tokens = np.random.default_rng(0).integers(0, vocab_size, size=length)
    return TokenDataset(tokens, context_length=context_length)


def test_get_lr_warmup_then_flat():
    base_lr = 1e-3
    warmup_steps = 10
    assert get_lr(0, warmup_steps, base_lr) == base_lr * (1 / 10)
    assert get_lr(9, warmup_steps, base_lr) == base_lr * (10 / 10)
    assert get_lr(10, warmup_steps, base_lr) == base_lr
    assert get_lr(1000, warmup_steps, base_lr) == base_lr


def test_get_lr_handles_zero_warmup():
    assert get_lr(0, 0, 1e-3) == 1e-3


def test_estimate_loss_does_not_update_weights():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)

    weights_before = model.lm_head.weight.clone()
    loss = estimate_loss(model, loader, num_batches=3, device="cpu")
    weights_after = model.lm_head.weight

    assert isinstance(loss, float)
    assert torch.equal(weights_before, weights_after)
    assert model.training  # restored to train mode afterwards


def test_train_model_runs_and_produces_checkpoint(tmp_path):
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt"))
    model = GPT(model_config)
    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)

    result = train_model(
        model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None
    )

    assert len(result["train_losses"]) == train_config.max_steps
    assert all(isinstance(x, float) for x in result["train_losses"])

    checkpoint_path = tmp_path / "ckpt" / "latest.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert checkpoint["model_config"].vocab_size == model_config.vocab_size
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["step"] == train_config.max_steps


def test_train_model_saves_tokenizer_when_given_one(tmp_path):
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt"))
    model = GPT(model_config)
    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)

    tokenizer = BPETokenizer()
    tokenizer.train("hello world, this is a tiny corpus for testing.", vocab_size=260)

    result = train_model(
        model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None, tokenizer=tokenizer
    )

    tokenizer_path = tmp_path / "ckpt" / "tokenizer.json"
    assert tokenizer_path.exists()
    assert result["tokenizer_path"] == str(tokenizer_path)

    loaded = BPETokenizer.load(tokenizer_path)
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab == tokenizer.vocab


def test_train_model_without_tokenizer_saves_no_tokenizer_file(tmp_path):
    """tokenizer is optional -- omitting it must not create a tokenizer.json
    or add a tokenizer_path key, to avoid silently implying one was saved.
    """
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt"))
    model = GPT(model_config)
    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)

    result = train_model(model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None)

    assert not (tmp_path / "ckpt" / "tokenizer.json").exists()
    assert "tokenizer_path" not in result


def test_weights_change_after_a_training_step(tmp_path):
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(
        max_steps=1, checkpoint_dir=str(tmp_path / "ckpt")
    )
    model = GPT(model_config)
    weight_before = model.lm_head.weight.clone()

    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)
    train_model(model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None)

    assert not torch.equal(weight_before, model.lm_head.weight)


def test_loss_decreases_on_tiny_overfit_run(tmp_path):
    """A small model training on a tiny repeating token stream should overfit
    quickly -- loss at the end should be clearly lower than at the start.
    This exercises the full step: forward -> loss -> backward -> optimizer step.
    """
    model_config = _tiny_model_config(vocab_size=10, context_length=8, n_layer=2, n_embd=32)
    train_config = _tiny_train_config(
        batch_size=4,
        learning_rate=5e-3,
        warmup_steps=5,
        max_steps=100,
        eval_every=100,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    model = GPT(model_config)

    # a short repeating pattern is easy to overfit
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tokens = np.array(pattern * 20)
    train_dataset = TokenDataset(tokens, context_length=model_config.context_length)
    val_dataset = TokenDataset(tokens, context_length=model_config.context_length)

    result = train_model(
        model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None
    )
    losses = result["train_losses"]

    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 100
  context_length: 16
  n_layer: 1
  n_head: 1
  n_embd: 8
  dropout: 0.0
train:
  batch_size: 2
  learning_rate: 0.001
  warmup_steps: 1
  max_steps: 2
  grad_clip: 1.0
  eval_every: 2
  checkpoint_dir: ckpt/
  checkpoint_every: 2
"""
    )
    model_config, train_config = load_config(config_path)
    assert model_config.vocab_size == 100
    assert train_config.max_steps == 2


def test_small_config_loads():
    """configs/small.yaml (this project's real config) should parse cleanly."""
    model_config, train_config = load_config("configs/small.yaml")
    assert model_config.vocab_size == 8000
    assert train_config.batch_size == 32


def test_load_config_coerces_bare_exponent_learning_rate(tmp_path):
    """PyYAML parses '3e-4' (no decimal point) as a string, not a float --
    load_config must coerce it, or the optimizer breaks on a bare-exponent
    learning_rate like the one in configs/small.yaml.
    """
    config_path = tmp_path / "bare_exponent.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 100
  context_length: 16
  n_layer: 1
  n_head: 1
  n_embd: 8
  dropout: 0.0
train:
  batch_size: 2
  learning_rate: 3e-4
  warmup_steps: 1
  max_steps: 2
  grad_clip: 1.0
  eval_every: 2
  checkpoint_dir: ckpt/
  checkpoint_every: 2
"""
    )
    _, train_config = load_config(config_path)
    assert isinstance(train_config.learning_rate, float)
    assert train_config.learning_rate == 3e-4


def test_checkpoint_written_periodically_not_only_at_end(tmp_path):
    """checkpoint_every should cause latest.pt to exist even before max_steps
    is reached -- this is what bounds lost work on a crash.
    """
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(
        max_steps=10, checkpoint_every=3, checkpoint_dir=str(tmp_path / "ckpt")
    )
    model = GPT(model_config)
    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)

    checkpoint_path = tmp_path / "ckpt" / "latest.pt"
    seen_steps = []

    orig_save = torch.save

    def _spy_save(obj, path, *a, **kw):
        seen_steps.append(obj["step"])
        return orig_save(obj, path, *a, **kw)

    import llm_from_scratch.train.loop as loop_module

    old_torch_save = loop_module.torch.save
    loop_module.torch.save = _spy_save
    try:
        train_model(model, train_dataset, val_dataset, train_config, log_fn=lambda _msg: None)
    finally:
        loop_module.torch.save = old_torch_save

    assert seen_steps == [3, 6, 9, 10]
    assert checkpoint_path.exists()


def test_load_checkpoint_for_resume_missing_file_raises(tmp_path):
    model_config = _tiny_model_config()
    with pytest.raises(FileNotFoundError):
        load_checkpoint_for_resume(tmp_path / "does_not_exist", model_config)


def test_load_checkpoint_for_resume_rejects_pre_milestone_checkpoint(tmp_path):
    """A checkpoint saved before this milestone (no optimizer_state_dict/step)
    must fail loudly, not silently resume with step=0 / a fresh optimizer.
    """
    model_config = _tiny_model_config()
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    torch.save(
        {"model_state_dict": GPT(model_config).state_dict(), "model_config": model_config},
        checkpoint_dir / "latest.pt",
    )

    with pytest.raises(ValueError, match="missing"):
        load_checkpoint_for_resume(checkpoint_dir, model_config)


def test_load_checkpoint_for_resume_rejects_mismatched_model_config(tmp_path):
    model_config = _tiny_model_config()
    other_config = _tiny_model_config(n_embd=32)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "model_state_dict": GPT(other_config).state_dict(),
            "model_config": other_config,
            "optimizer_state_dict": {},
            "step": 5,
        },
        checkpoint_dir / "latest.pt",
    )

    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_for_resume(checkpoint_dir, model_config)


def test_resume_continues_from_saved_step_not_zero(tmp_path):
    """The core resume guarantee: interrupt at step N, resume, and training
    continues at step N -- not step 0, and not by rerunning 0..max_steps.
    """
    model_config = _tiny_model_config()
    checkpoint_dir = tmp_path / "ckpt"

    first_config = _tiny_train_config(
        max_steps=4, checkpoint_every=4, checkpoint_dir=str(checkpoint_dir)
    )
    model = GPT(model_config)
    train_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)

    train_model(model, train_dataset, val_dataset, first_config, log_fn=lambda _msg: None)

    checkpoint = load_checkpoint_for_resume(checkpoint_dir, model_config)
    assert checkpoint["step"] == 4

    resumed_model = GPT(model_config)
    resumed_model.load_state_dict(checkpoint["model_state_dict"])

    second_config = _tiny_train_config(
        max_steps=10, checkpoint_every=10, checkpoint_dir=str(checkpoint_dir)
    )
    result = train_model(
        resumed_model,
        train_dataset,
        val_dataset,
        second_config,
        log_fn=lambda _msg: None,
        start_step=checkpoint["step"],
        optimizer_state_dict=checkpoint["optimizer_state_dict"],
    )

    # steps 4..9 = 6 more training steps, not 10
    assert len(result["train_losses"]) == 6

    final_checkpoint = torch.load(checkpoint_dir / "latest.pt", weights_only=False)
    assert final_checkpoint["step"] == 10
