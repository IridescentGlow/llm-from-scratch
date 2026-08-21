"""
Entry point for text generation from a checkpoint.
Usage: python scripts/generate.py --checkpoint checkpoints/latest.pt \
    --config configs/small.yaml --prompt "..." \
    [--max-new-tokens 50] [--temperature 0.0]
"""
import argparse
from pathlib import Path

import yaml

from llm_from_scratch.finetune import load_pretrained_model
from llm_from_scratch.generate import generate
from llm_from_scratch.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model = load_pretrained_model(args.checkpoint)

    with open(args.config) as f:
        data_config = yaml.safe_load(f)["data"]

    raw_path = Path(data_config["raw_path"])
    text_files = sorted(raw_path.glob("*.txt"))
    if not text_files:
        raise SystemExit(
            f"No .txt files found in {raw_path}. Same corpus used for training is needed."
        )
    corpus = "\n".join(f.read_text() for f in text_files)

    # Tokenizer has no persistence yet (Stage 1) -- retraining on the same
    # corpus + vocab_size deterministically reproduces the same tokenizer
    # used at training time. Only valid if data/raw hasn't changed since the
    # checkpoint was trained. Same known limitation as scripts/evaluate.py.
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=model.config.vocab_size)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
