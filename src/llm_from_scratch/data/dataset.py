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
    """

    def __init__(self, token_ids: np.ndarray, context_length: int) -> None:
        if len(token_ids) <= context_length:
            raise ValueError(
                f"token stream (len={len(token_ids)}) must be longer than "
                f"context_length ({context_length})"
            )
        self.token_ids = token_ids
        self.context_length = context_length

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
