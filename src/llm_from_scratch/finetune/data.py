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


def _prompt_prefix(example: InstructionExample) -> str:
    """The part of `format_example`'s output that's given, not generated --
    everything up to and including the `Response: ` label. See
    docs/finetune-loss-masking.md."""
    return f"Instruction: {example.instruction}\nResponse: "


def build_masked_token_ids(
    examples: list[InstructionExample], tokenizer: BPETokenizer
) -> tuple[list[int], list[bool]]:
    """Like `build_token_ids`, but also returns a same-length loss mask:
    False for prompt tokens, True for response tokens and EOS.

    See docs/finetune-loss-masking.md for the full design and worked
    example. Per example, the prompt/response split point is found by
    encoding the prompt prefix on its own and taking `len(encode(prefix))`
    as the boundary -- this project's word-boundary pretokenization means
    the prefix's tokens come out the same whether encoded alone or as the
    start of the full formatted string in practice, though this isn't
    guaranteed for every possible tokenizer/string combination (documented
    limitation, not a silent assumption).

    Raises ValueError if `tokenizer` has no EOS token, same as
    `build_token_ids`.
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
    loss_mask: list[bool] = []
    for example in examples:
        prefix_ids = tokenizer.encode(_prompt_prefix(example))
        full_ids = tokenizer.encode(format_example(example))
        prefix_len = len(prefix_ids)

        token_ids.extend(full_ids)
        loss_mask.extend([False] * prefix_len)
        loss_mask.extend([True] * (len(full_ids) - prefix_len))

        token_ids.append(tokenizer.eos_token_id)
        loss_mask.append(True)
    return token_ids, loss_mask


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
