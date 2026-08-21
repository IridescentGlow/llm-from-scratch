"""
Stage 7 — Generation / Inference.
See docs/07-generation.md before implementing here.
Expected surface: prompt -> token ids -> generation loop -> text, reusing
Stage 3's GPT.generate and Stage 6's load_pretrained_model.
"""

from .inference import generate

__all__ = ["generate"]
