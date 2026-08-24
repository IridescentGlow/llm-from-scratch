"""Windowing + batching. See docs/02-data-pipeline.md for the concept walkthrough."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """Slides a fixed-size window over a token stream.

    Each item `i` is one training example:
    - input:  token_ids[i : i + context_length]
    - target: token_ids[i+1 : i + context_length + 1]  (input shifted by one)

    `loss_mask`, if given, is a same-length companion array to `token_ids`:
    `loss_mask[i]` is truthy if `token_ids[i]` should count as a loss target
    when it's predicted, falsy if it shouldn't (e.g. a fine-tuning prompt
    token). When a target position's mask value is falsy, that position's
    target id is set to -100 -- PyTorch's `F.cross_entropy` default
    `ignore_index`, so it contributes no loss and no gradient with no
    change needed to the model or loss call. See
    docs/finetune-loss-masking.md. Default `None` leaves every target as
    a normal loss target, exactly as before this existed (pretraining and
    evaluation are unaffected).
    """

    def __init__(
        self,
        token_ids: np.ndarray,
        context_length: int,
        loss_mask: np.ndarray | None = None,
    ) -> None:
        if len(token_ids) <= context_length:
            raise ValueError(
                f"token stream (len={len(token_ids)}) must be longer than "
                f"context_length ({context_length})"
            )
        if loss_mask is not None and len(loss_mask) != len(token_ids):
            raise ValueError(
                f"loss_mask length ({len(loss_mask)}) must match token_ids "
                f"length ({len(token_ids)})"
            )
        self.token_ids = token_ids
        self.context_length = context_length
        self.loss_mask = loss_mask

    def __len__(self) -> int:
        # number of valid window start positions
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.token_ids[idx : idx + self.context_length + 1]
        # np.array() copies out of the memmap here, so the returned tensors
        # own their memory instead of aliasing the on-disk file.
        window = np.array(window, dtype=np.int64)
        input_ids = torch.from_numpy(window[:-1])
        target_ids = torch.from_numpy(window[1:])

        if self.loss_mask is not None:
            # window[:-1] and window[1:] are overlapping views into the same
            # buffer, so input_ids and target_ids alias each other's memory
            # -- masked_fill (out-of-place) avoids corrupting input_ids the
            # way an in-place target_ids[...] = -100 assignment would.
            mask_window = self.loss_mask[idx + 1 : idx + self.context_length + 1]
            mask_window = torch.from_numpy(np.array(mask_window, dtype=bool))
            target_ids = target_ids.masked_fill(~mask_window, -100)

        return input_ids, target_ids


def get_dataloader(
    dataset: TokenDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Wrap a TokenDataset into batches of shape (batch_size, context_length).

    `shuffle=True` (the default, used for training) draws window start
    positions in random order each epoch. Validation loops should pass
    `shuffle=False` for deterministic, repeatable batches.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
