"""
Entry point for instruction fine-tuning on top of a pretrained checkpoint.
Usage: python scripts/finetune.py --checkpoint checkpoints/latest.pt --config configs/finetune.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from llm_from_scratch.data import TokenDataset, load_token_ids, train_val_split, write_token_ids
from llm_from_scratch.device import resolve_device
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.finetune import (
    build_corpus,
    load_examples_jsonl,
    load_pretrained_model,
    load_tokenizer_for_checkpoint,
)
from llm_from_scratch.train import TrainConfig, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="Instruction: What is the capital of France?\nResponse:")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = load_pretrained_model(args.checkpoint)
    model.to(device)
    model_config = model.config

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    train_raw = dict(raw["train"])
    train_raw["learning_rate"] = float(train_raw["learning_rate"])
    train_config = TrainConfig(**train_raw)
    data_config = raw["data"]

    examples = load_examples_jsonl(data_config["examples_path"])
    corpus = build_corpus(examples)

    # Load the exact tokenizer this checkpoint was pretrained with -- do NOT
    # retrain a new one on the (small, differently-distributed) fine-tuning
    # corpus. Retraining here used to silently give token ids a different
    # meaning than the ones the pretrained embedding table learned (see
    # docs/01-tokenization.md, "Tokenizer persistence", and
    # docs/06-finetuning.md).
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint)
    token_ids = tokenizer.encode(corpus)

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    tokens_path = processed_path / "tokens.bin"
    write_token_ids(token_ids, tokens_path)
    all_tokens = load_token_ids(tokens_path)

    train_tokens, val_tokens = train_val_split(all_tokens, data_config["train_split"])
    train_dataset = TokenDataset(train_tokens, context_length=model_config.context_length)
    val_dataset = TokenDataset(val_tokens, context_length=model_config.context_length)

    before = evaluate_model(model, val_dataset, batch_size=train_config.batch_size, device=device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    before_text = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=20)[0].tolist())

    print(f"before fine-tuning: val_loss={before['loss']:.4f} perplexity={before['perplexity']:.2f}")
    print(f"before fine-tuning generation:\n{before_text}\n")

    result = train_model(
        model, train_dataset, val_dataset, train_config, device=device, tokenizer=tokenizer
    )

    after = evaluate_model(model, val_dataset, batch_size=train_config.batch_size, device=device)
    after_text = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=20)[0].tolist())

    print(f"after fine-tuning: val_loss={after['loss']:.4f} perplexity={after['perplexity']:.2f}")
    print(f"after fine-tuning generation:\n{after_text}\n")
    print(f"Fine-tuned checkpoint saved to {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
