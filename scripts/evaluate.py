"""
Entry point for evaluation (val loss/perplexity).
Usage: python scripts/evaluate.py --checkpoint checkpoints/latest.pt --config configs/small.yaml
"""
import argparse
from pathlib import Path

import yaml

from llm_from_scratch.data import TokenDataset, ensure_token_cache, load_token_ids, train_val_split
from llm_from_scratch.device import resolve_device
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.finetune import load_pretrained_model, load_tokenizer_for_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Reuses the same shared, safe (weights_only=True) checkpoint loader as
    # scripts/generate.py and scripts/finetune.py -- see
    # docs/checkpoint-format.md. Loads onto CPU first regardless of what
    # device the checkpoint was saved from; evaluate_model moves the model
    # to `device` itself. See docs/device-support.md.
    model = load_pretrained_model(args.checkpoint)
    model_config = model.config

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

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    tokens_path = processed_path / "tokens.bin"
    meta_path = processed_path / "tokens.meta.json"
    # See docs/token-cache.md -- reuses tokens.bin when it already holds
    # this corpus encoded with this exact tokenizer.
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
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
