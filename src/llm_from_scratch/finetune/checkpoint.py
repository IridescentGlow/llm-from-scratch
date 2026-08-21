"""Loading a pretrained checkpoint to continue training from. See docs/06-finetuning.md."""

from __future__ import annotations

from pathlib import Path

import torch

from llm_from_scratch.model import GPT


def load_pretrained_model(checkpoint_path: str | Path) -> GPT:
    """Rebuild a GPT from a Stage 4 checkpoint (model_config + model_state_dict)."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = GPT(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
