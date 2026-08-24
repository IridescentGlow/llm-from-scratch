"""Token storage on disk. See docs/02-data-pipeline.md for the concept walkthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# uint16 covers vocab sizes up to 65,536 -- this project's configs stay well
# under that (configs/small.yaml uses 8000), and it halves storage vs int32.
TOKEN_DTYPE = np.uint16

# A loss mask only ever needs 0/1, so a single byte per position is enough --
# no need for TOKEN_DTYPE's range. See docs/finetune-loss-masking.md.
MASK_DTYPE = np.uint8


def write_token_ids(token_ids: list[int], path: str | Path) -> None:
    """Write token ids to `path` as a flat binary file of TOKEN_DTYPE integers."""
    array = np.array(token_ids, dtype=TOKEN_DTYPE)
    array.tofile(path)


def load_token_ids(path: str | Path) -> np.memmap:
    """Memory-map a token file written by `write_token_ids`.

    Returns a read-only memmap: indexing/slicing it reads from disk on
    demand instead of loading the whole file into RAM.
    """
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def write_loss_mask(loss_mask: list[bool], path: str | Path) -> None:
    """Write a loss mask to `path`, aligned position-for-position with a
    token id file written by `write_token_ids`. See
    docs/finetune-loss-masking.md."""
    array = np.array(loss_mask, dtype=MASK_DTYPE)
    array.tofile(path)


def load_loss_mask(path: str | Path) -> np.memmap:
    """Memory-map a loss mask file written by `write_loss_mask`."""
    return np.memmap(path, dtype=MASK_DTYPE, mode="r")


def train_val_split(
    token_ids: np.ndarray, train_split: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split a token array by position: first `train_split` fraction is train."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1 (exclusive)")
    split_index = int(len(token_ids) * train_split)
    return token_ids[:split_index], token_ids[split_index:]
