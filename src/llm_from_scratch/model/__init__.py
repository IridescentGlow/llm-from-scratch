"""
Stage 3 — Architecture.
See docs/03-architecture.md before implementing here.
Expected surface: a GPT-style nn.Module built from hand-written attention blocks.
"""

from .gpt import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
