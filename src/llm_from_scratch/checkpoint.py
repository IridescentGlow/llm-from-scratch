"""Shared, safe checkpoint loading. See docs/checkpoint-format.md.

`load_pretrained_model` (finetune/checkpoint.py) and
`load_checkpoint_for_resume` (train/loop.py) both need to read a
checkpoint's `model_state_dict`/`model_config`/etc. off disk -- this
module is the one place that actually calls `torch.load`, so both go
through the same `weights_only=True` load and the same legacy-checkpoint
error message.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from llm_from_scratch.model import GPTConfig


def load_checkpoint_dict(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint file into a dict, safely.

    Always uses `weights_only=True` -- there is no fallback to unrestricted
    (unsafe) unpickling, even for a checkpoint that fails to load this
    way (see below). `model_config` is stored on disk as a plain dict
    (JSON-compatible data, not a pickled `GPTConfig` object -- see
    docs/checkpoint-format.md) and is reconstructed into a real
    `GPTConfig` here, so every caller gets back the same shape of object
    `load()` always returned before this milestone.

    Raises FileNotFoundError if `path` doesn't exist. Raises ValueError,
    with an explicit explanation, for a checkpoint saved before this
    milestone -- one whose `model_config` was pickled as a `GPTConfig`
    object rather than a plain dict, which `weights_only=True` refuses to
    unpickle. That refusal (a `pickle.UnpicklingError`) is the actual
    detection mechanism for "this is a legacy checkpoint" -- no separate
    version field is needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}.")

    try:
        checkpoint = torch.load(path, weights_only=True, map_location=map_location)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"Checkpoint at {path} could not be loaded safely (weights_only=True "
            "rejected it). This checkpoint predates safe checkpoint serialization "
            "-- it likely has model_config pickled as a GPTConfig object instead "
            "of a plain dict -- and cannot be loaded. Regenerate it with the "
            "current scripts/train.py. See docs/checkpoint-format.md."
        ) from e

    if isinstance(checkpoint.get("model_config"), dict):
        checkpoint["model_config"] = GPTConfig(**checkpoint["model_config"])

    return checkpoint
