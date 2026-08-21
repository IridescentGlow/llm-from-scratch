"""Evaluation: validation loss and perplexity. See docs/05-evaluation.md."""

from __future__ import annotations

import math

import torch

from llm_from_scratch.data import TokenDataset, get_dataloader
from llm_from_scratch.model import GPT


@torch.no_grad()
def evaluate_model(
    model: GPT,
    dataset: TokenDataset,
    batch_size: int,
    device: str = "cpu",
    max_batches: int | None = None,
) -> dict:
    """Compute validation loss and perplexity over `dataset`.

    Runs forward passes only (no backward pass, no optimizer step) with the
    model in eval() mode, over every batch in `dataset` in a fixed order
    (or just the first `max_batches`, if given) -- deterministic, unlike
    training's shuffled sampling.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    loader = get_dataloader(dataset, batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0
    num_tokens = 0

    for batch_idx, (input_ids, target_ids) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        _, loss = model(input_ids, target_ids)

        total_loss += loss.item()
        num_batches += 1
        num_tokens += target_ids.numel()

    model.train(was_training)

    if num_batches == 0:
        raise ValueError("dataset produced no batches to evaluate")

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_batches": num_batches,
        "num_tokens": num_tokens,
    }
