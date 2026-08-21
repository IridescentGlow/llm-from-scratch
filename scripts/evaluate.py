"""
Entry point for evaluation (val loss/perplexity).
Usage: python scripts/evaluate.py --checkpoint checkpoints/latest.pt --config configs/small.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from llm_from_scratch.data import TokenDataset, load_token_ids, train_val_split, write_token_ids
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.model import GPT
from llm_from_scratch.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model_config = checkpoint["model_config"]
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    data_config = raw["data"]

    raw_path = Path(data_config["raw_path"])
    text_files = sorted(raw_path.glob("*.txt"))
    if not text_files:
        raise SystemExit(
            f"No .txt files found in {raw_path}. Same corpus used for training is needed."
        )
    corpus = "\n".join(f.read_text() for f in text_files)

    # Tokenizer has no persistence yet (Stage 1) -- retraining on the same
    # corpus + vocab_size deterministically reproduces the same token
    # stream used at training time. Only valid if data/raw hasn't changed
    # since the checkpoint was trained.
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=model_config.vocab_size)
    token_ids = tokenizer.encode(corpus)

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    tokens_path = processed_path / "tokens.bin"
    write_token_ids(token_ids, tokens_path)
    all_tokens = load_token_ids(tokens_path)

    _, val_tokens = train_val_split(all_tokens, data_config["train_split"])
    val_dataset = TokenDataset(val_tokens, context_length=model_config.context_length)

    result = evaluate_model(model, val_dataset, batch_size=raw["train"]["batch_size"])

    print(f"val_loss:   {result['loss']:.4f}")
    print(f"perplexity: {result['perplexity']:.2f}")
    print(f"batches:    {result['num_batches']}")
    print(f"tokens:     {result['num_tokens']}")


if __name__ == "__main__":
    main()
