"""Instruction/response formatting. See docs/06-finetuning.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
