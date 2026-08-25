"""
Stage 4 — Pretraining loop.
See docs/04-pretraining.md before implementing here.
Expected surface: a plain PyTorch training loop with checkpointing + logging.
"""

from .loop import (
    TrainConfig,
    configure_optimizer,
    estimate_loss,
    get_lr,
    load_checkpoint_for_resume,
    load_config,
    train_model,
)

__all__ = [
    "TrainConfig",
    "configure_optimizer",
    "estimate_loss",
    "get_lr",
    "load_checkpoint_for_resume",
    "load_config",
    "train_model",
]
