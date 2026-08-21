"""Prompt-to-text generation. See docs/07-generation.md."""

from __future__ import annotations

import torch

from llm_from_scratch.model import GPT
from llm_from_scratch.tokenizer import BPETokenizer


def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    device: str = "cpu",
) -> str:
    """Encode `prompt`, run GPT.generate, decode the result back to text."""
    model.to(device)
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out_ids[0].tolist())
