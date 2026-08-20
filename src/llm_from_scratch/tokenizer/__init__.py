"""
Stage 1 — Tokenization.
See docs/01-tokenization.md before implementing here.
Expected surface: a BPE tokenizer with .train(), .encode(), .decode().
"""

from .bpe import BPETokenizer

__all__ = ["BPETokenizer"]
