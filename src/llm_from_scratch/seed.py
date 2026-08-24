"""Reproducible-run seeding. See docs/reproducibility.md."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG this project draws from: Python `random`, NumPy, and
    PyTorch (CPU and CUDA). Call this before constructing any model or
    dataloader, so model init, dropout, and DataLoader shuffling all become
    reproducible given the same seed, environment, and device. See
    docs/reproducibility.md for exactly what this does and does not
    guarantee (in particular: not exact RNG-stream continuation across
    --resume, and not bit-exact cross-device/cross-version reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no CUDA device is present
