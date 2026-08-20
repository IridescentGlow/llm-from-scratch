"""Pretraining loop. See docs/04-pretraining.md for the concept walkthrough."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import yaml

from llm_from_scratch.data import TokenDataset, get_dataloader
from llm_from_scratch.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    eval_every: int
    checkpoint_dir: str


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
    train_config = TrainConfig(**train_raw)
    return model_config, train_config


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup from ~0 to base_lr over warmup_steps, then flat.

    See the "Learning rate and warmup" section of docs/04-pretraining.md.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / warmup_steps


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
) -> dict:
    """Run the pretraining loop for `config.max_steps` steps.

    One step = one batch: forward pass -> loss -> backward pass -> optimizer
    step, exactly as described in docs/04-pretraining.md. Returns a dict with
    per-step train losses and the checkpoint path.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, config.batch_size, shuffle=False)
    train_iter = iter(train_loader)

    train_losses: list[float] = []

    for step in range(config.max_steps):
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        lr = get_lr(step, config.warmup_steps, config.learning_rate)
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

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model.config},
        checkpoint_path,
    )

    return {"train_losses": train_losses, "checkpoint_path": str(checkpoint_path)}
