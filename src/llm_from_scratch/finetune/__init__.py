"""
Stage 6 — Fine-tuning.
See docs/06-finetuning.md before implementing here.
Expected surface: instruction-tuning pass on top of a pretrained checkpoint.
"""

from .checkpoint import load_pretrained_model, load_tokenizer_for_checkpoint
from .data import (
    InstructionExample,
    build_corpus,
    build_masked_token_ids,
    build_token_ids,
    format_example,
    load_examples_jsonl,
)

__all__ = [
    "InstructionExample",
    "build_corpus",
    "build_masked_token_ids",
    "build_token_ids",
    "format_example",
    "load_examples_jsonl",
    "load_pretrained_model",
    "load_tokenizer_for_checkpoint",
]
