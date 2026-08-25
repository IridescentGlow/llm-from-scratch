"""Pretraining loop. See docs/04-pretraining.md for the concept walkthrough."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import yaml

from llm_from_scratch.checkpoint import load_checkpoint_dict
from llm_from_scratch.data import TokenDataset, get_dataloader
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    min_lr: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    eval_every: int
    checkpoint_dir: str
    checkpoint_every: int
    weight_decay: float


def load_config(path: str | Path) -> tuple[GPTConfig, TrainConfig]:
    """Read `configs/*.yaml` into (GPTConfig, TrainConfig). See configs/small.yaml."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    model_config = GPTConfig(**raw["model"])
    train_raw = dict(raw["train"])
    # PyYAML's default resolver only recognizes floats with a decimal point
    # (e.g. "3.0e-4"), not bare exponent notation like "3e-4" -- it parses
    # the latter as a string. Coerce explicitly rather than relying on
    # every config file to use the decimal-point form.
    train_raw["learning_rate"] = float(train_raw["learning_rate"])
    train_raw["min_lr"] = float(train_raw["min_lr"])
    train_config = TrainConfig(**train_raw)
    return model_config, train_config


def load_checkpoint_for_resume(checkpoint_dir: str | Path, model_config: GPTConfig) -> dict:
    """Load `latest.pt` for `--resume`, validating it's actually resumable.

    Raises FileNotFoundError if no checkpoint exists at checkpoint_dir, and
    ValueError if it's missing optimizer_state_dict/step (saved before this
    milestone existed), was saved with a different model_config, or predates
    safe checkpoint serialization (docs/checkpoint-format.md) -- any of
    these would make resuming silently wrong (or unsafe) rather than just
    inconvenient. See docs/checkpoint-resume.md.
    """
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    try:
        checkpoint = load_checkpoint_dict(checkpoint_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"--resume was given but no checkpoint found at {checkpoint_path}."
        ) from None

    missing = [k for k in ("optimizer_state_dict", "step") if k not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing {missing} -- it predates "
            "mid-training checkpointing/resume support and cannot be resumed from. "
            "See docs/checkpoint-resume.md."
        )
    if checkpoint["model_config"] != model_config:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} was saved with model_config "
            f"{checkpoint['model_config']}, which does not match the model_config "
            f"passed in ({model_config}). Refusing to resume from a mismatched "
            "architecture."
        )
    return checkpoint


def configure_optimizer(model: GPT, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    """Build AdamW with weight decay applied only to weight matrices.

    Decaying a bias or a LayerNorm gain toward zero doesn't fight
    overfitting the way it does for a weight matrix -- it just biases the
    model away from a shift/scale it may genuinely need. Standard practice
    (see nanoGPT) is to split parameters by shape: anything with 2+ dims
    (Linear/Embedding weights) gets `weight_decay`; anything 1-D (biases,
    LayerNorm weights) gets none. Previously every parameter went through
    a single `torch.optim.AdamW(model.parameters(), lr=...)` call, which
    applied AdamW's default `weight_decay=0.01` uniformly.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup from ~0 to base_lr, then cosine decay down to min_lr.

    See docs/lr-decay.md for the full design and worked example.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataloader, num_batches: int, device: str) -> float:
    """Average loss over up to `num_batches`, without updating weights."""
    was_training = model.training
    model.eval()

    losses = []
    for i, (input_ids, target_ids) in enumerate(dataloader):
        if i >= num_batches:
            break
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        _, loss = model(input_ids, target_ids)
        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("nan")


def train_model(
    model: GPT,
    train_dataset: TokenDataset,
    val_dataset: TokenDataset,
    config: TrainConfig,
    device: str = "cpu",
    log_every: int = 1,
    log_fn: Callable[[str], None] = print,
    tokenizer: BPETokenizer | None = None,
    start_step: int = 0,
    optimizer_state_dict: dict | None = None,
) -> dict:
    """Run the pretraining loop from `start_step` to `config.max_steps`.

    One step = one batch: forward pass -> loss -> backward pass -> optimizer
    step, exactly as described in docs/04-pretraining.md. Returns a dict with
    per-step train losses and the checkpoint path.

    If `tokenizer` is given, it's saved as `tokenizer.json` next to the
    checkpoint (see docs/01-tokenization.md, "Tokenizer persistence") so the
    checkpoint directory is a self-contained unit: weights + config +
    the exact tokenizer that produced this run's training data.

    `start_step` and `optimizer_state_dict` support resuming mid-run (see
    docs/checkpoint-resume.md): pass the values from
    `load_checkpoint_for_resume` to continue the learning-rate schedule and
    AdamW's per-weight momentum/variance from where a prior run left off,
    instead of restarting both from scratch. `model` is expected to already
    have the resumed weights loaded into it by the caller (mirroring how a
    fresh `model` is already constructed by the caller too).
    """
    model.to(device)
    model.train()
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, config.batch_size, shuffle=False)
    train_iter = iter(train_loader)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"

    def save_checkpoint() -> Path:
        # Write to a temp file first, then atomically rename it into place
        # (os.replace) -- a crash mid-write leaves the old latest.pt intact
        # instead of truncated. See docs/checkpoint-atomicity.md.
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                # Plain dict, not the GPTConfig object -- so the checkpoint
                # can be loaded with weights_only=True. See
                # docs/checkpoint-format.md.
                "model_config": asdict(model.config),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step + 1,
            },
            tmp_path,
        )
        os.replace(tmp_path, checkpoint_path)
        return checkpoint_path

    train_losses: list[float] = []

    for step in range(start_step, config.max_steps):
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        lr = get_lr(step, config.warmup_steps, config.max_steps, config.learning_rate, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        _, loss = model(input_ids, target_ids)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        is_last_step = step == config.max_steps - 1
        if (step + 1) % config.eval_every == 0 or is_last_step:
            val_loss = estimate_loss(model, val_loader, num_batches=5, device=device)
            if (step + 1) % log_every == 0 or is_last_step:
                log_fn(
                    f"step {step + 1}/{config.max_steps} "
                    f"train_loss={loss.item():.4f} val_loss={val_loss:.4f} lr={lr:.2e}"
                )

        if (step + 1) % config.checkpoint_every == 0 or is_last_step:
            save_checkpoint()

    result = {"train_losses": train_losses, "checkpoint_path": str(checkpoint_path)}
    if tokenizer is not None:
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        result["tokenizer_path"] = str(tokenizer_path)
    return result
