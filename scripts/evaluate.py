"""
Entry point for evaluation (val loss/perplexity).
Usage: python scripts/evaluate.py --checkpoint checkpoints/latest.pt --config configs/small.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from llm_from_scratch.data import TokenDataset, load_token_ids, train_val_split, write_token_ids
from llm_from_scratch.device import resolve_device
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.finetune import load_tokenizer_for_checkpoint
from llm_from_scratch.model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # map_location="cpu" so a checkpoint saved from a GPU run still loads on
    # a CPU-only machine -- see docs/device-support.md. evaluate_model moves
    # the model to `device` itself.
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
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

    # Loads the exact tokenizer train_model saved alongside this checkpoint
    # -- see docs/01-tokenization.md, "Tokenizer persistence". Raises a
    # clear error instead of retraining if this checkpoint predates
    # tokenizer persistence.
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint)
    token_ids = tokenizer.encode(corpus)

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    tokens_path = processed_path / "tokens.bin"
    write_token_ids(token_ids, tokens_path)
    all_tokens = load_token_ids(tokens_path)

    _, val_tokens = train_val_split(all_tokens, data_config["train_split"])
    val_dataset = TokenDataset(val_tokens, context_length=model_config.context_length)

    result = evaluate_model(
        model, val_dataset, batch_size=raw["train"]["batch_size"], device=device
    )

    print(f"val_loss:   {result['loss']:.4f}")
    print(f"perplexity: {result['perplexity']:.2f}")
    print(f"batches:    {result['num_batches']}")
    print(f"tokens:     {result['num_tokens']}")


if __name__ == "__main__":
    main()
