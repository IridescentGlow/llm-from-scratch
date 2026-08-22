"""Instruction/response formatting. See docs/06-finetuning.md and
docs/eos-generation-stopping.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_from_scratch.tokenizer import BPETokenizer

TEMPLATE = "Instruction: {instruction}\nResponse: {response}\n"


@dataclass
class InstructionExample:
    instruction: str
    response: str


def format_example(example: InstructionExample) -> str:
    """Flatten one (instruction, response) pair into the fixed training template."""
    return TEMPLATE.format(instruction=example.instruction, response=example.response)


def build_corpus(examples: list[InstructionExample]) -> str:
    """Join formatted examples into one text corpus, ready for the Stage 1/2 pipeline."""
    return "\n".join(format_example(example) for example in examples)


def build_token_ids(examples: list[InstructionExample], tokenizer: BPETokenizer) -> list[int]:
    """Encode each example separately and append EOS after each response.

    See docs/eos-generation-stopping.md. EOS can't be embedded in
    `build_corpus`'s text blob -- it isn't text, it has no bytes -- so each
    example is encoded on its own and the tokenizer's EOS id is appended to
    *that example's* token ids, before concatenating across examples. This
    teaches the model "stop right after a complete response," at the
    boundary between examples, rather than only at the very end of the
    whole fine-tuning corpus.

    Raises ValueError if `tokenizer` has no EOS token -- silently omitting
    it would produce a fine-tuning run that looks like it worked but never
    actually taught the model to stop. Fine-tune against a checkpoint
    pretrained with the current scripts/train.py (which adds EOS) instead.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError(
            "This checkpoint's tokenizer has no EOS token, so fine-tuning "
            "can't teach the model when to stop (see "
            "docs/eos-generation-stopping.md). Pretrain a new checkpoint "
            "with the current scripts/train.py, which adds one, then "
            "fine-tune from that instead."
        )
    token_ids: list[int] = []
    for example in examples:
        token_ids.extend(tokenizer.encode(format_example(example)))
        token_ids.append(tokenizer.eos_token_id)
    return token_ids


def load_examples_jsonl(path: str | Path) -> list[InstructionExample]:
    """Load instruction/response pairs from a JSONL file (one {"instruction", "response"} per line)."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(InstructionExample(instruction=row["instruction"], response=row["response"]))
    return examples
