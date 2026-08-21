"""
Stage 5 — Evaluation.
See docs/05-evaluation.md before implementing here.
Expected surface: validation loss/perplexity + sample generation for eyeballing.
"""

from .metrics import evaluate_model

__all__ = ["evaluate_model"]
