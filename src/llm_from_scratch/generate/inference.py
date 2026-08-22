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
    """Encode `prompt`, run GPT.generate, decode the result back to text.

    Stops early if the tokenizer has an EOS token and the model produces it
    (see docs/eos-generation-stopping.md). A tokenizer with no EOS token
    (e.g. one saved before this milestone) simply generates the full
    `max_new_tokens` as before -- no error, no invented id.
    """
    model.to(device)
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0].tolist())
