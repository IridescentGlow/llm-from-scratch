"""
Stage 2 — Data pipeline.
See docs/02-data-pipeline.md before implementing here.
Expected surface: a Dataset/DataLoader yielding (input_ids, target_ids) batches.
"""

from .dataset import TokenDataset, get_dataloader
from .tokens import (
    load_loss_mask,
    load_token_ids,
    train_val_split,
    write_loss_mask,
    write_token_ids,
)

__all__ = [
    "TokenDataset",
    "get_dataloader",
    "load_loss_mask",
    "load_token_ids",
    "train_val_split",
    "write_loss_mask",
    "write_token_ids",
]
