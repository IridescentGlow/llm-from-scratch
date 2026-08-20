"""
Entry point for pretraining. Thin — logic lives in src/llm_from_scratch/train/.
Usage: python scripts/train.py --config configs/small.yaml
"""
import argparse
from pathlib import Path

import yaml

from llm_from_scratch.data import TokenDataset, load_token_ids, train_val_split, write_token_ids
from llm_from_scratch.model import GPT
from llm_from_scratch.tokenizer import BPETokenizer
from llm_from_scratch.train import load_config, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    model_config, train_config = load_config(args.config)
    with open(args.config) as f:
        data_config = yaml.safe_load(f)["data"]

    raw_path = Path(data_config["raw_path"])
    text_files = sorted(raw_path.glob("*.txt"))
    if not text_files:
        raise SystemExit(
            f"No .txt files found in {raw_path}. Add training text there first."
        )
    corpus = "\n".join(f.read_text() for f in text_files)

    print(f"Training tokenizer on {len(corpus):,} characters...")
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=model_config.vocab_size)

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    token_ids = tokenizer.encode(corpus)
    tokens_path = processed_path / "tokens.bin"
    write_token_ids(token_ids, tokens_path)

    all_tokens = load_token_ids(tokens_path)
    train_tokens, val_tokens = train_val_split(all_tokens, data_config["train_split"])
    train_dataset = TokenDataset(train_tokens, context_length=model_config.context_length)
    val_dataset = TokenDataset(val_tokens, context_length=model_config.context_length)

    model = GPT(model_config)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    result = train_model(model, train_dataset, val_dataset, train_config)
    print(f"Checkpoint saved to {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
