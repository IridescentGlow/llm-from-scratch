"""Device selection. See docs/device-support.md."""

from __future__ import annotations

import torch

_VALID_DEVICES = ("cpu", "cuda", "mps")


def resolve_device(requested: str | None = None) -> str:
    """Resolve a device string to actually use.

    If `requested` is None, auto-detect: CUDA if available, else MPS if
    available, else CPU. If `requested` is given, use exactly that device,
    or raise a clear error if it isn't available -- never silently fall
    back to something else. See docs/device-support.md, "Target behavior".
    """
    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested not in _VALID_DEVICES:
        raise ValueError(
            f"Unknown device '{requested}'. Choose from: {', '.join(_VALID_DEVICES)}."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but no CUDA GPU is available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device 'mps' but no MPS (Apple GPU) is available.")
    return requested
