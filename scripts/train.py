"""
Entry point for pretraining. Thin — logic lives in src/llm_from_scratch/train/.
Usage: python scripts/train.py --config configs/small.yaml
"""
import argparse
from pathlib import Path

import yaml

from llm_from_scratch.data import (
    TokenDataset,
    ensure_token_cache,
    load_token_ids,
    train_val_split,
)
from llm_from_scratch.device import resolve_device
from llm_from_scratch.finetune.checkpoint import load_tokenizer_for_checkpoint
from llm_from_scratch.model import GPT
from llm_from_scratch.seed import set_seed
from llm_from_scratch.tokenizer import BPETokenizer
from llm_from_scratch.train import load_checkpoint_for_resume, load_config, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from checkpoint_dir/latest.pt (model + optimizer state + "
            "step) instead of starting a fresh run. See docs/checkpoint-resume.md."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for reproducible initialization/dropout/shuffling. Omit for "
            "today's unseeded behavior. See docs/reproducibility.md."
        ),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.seed is not None:
        set_seed(args.seed)

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

    if args.resume:
        # Reuse the exact tokenizer this checkpoint was pretrained with,
        # rather than retraining BPE on whatever data/raw/ contains right
        # now -- retraining could silently produce a different id-to-string
        # mapping (different corpus content, file ordering, or vocab_size)
        # than the one the checkpoint's embedding table was trained under.
        # See docs/resume-tokenizer-consistency.md.
        checkpoint_path = Path(train_config.checkpoint_dir) / "latest.pt"
        tokenizer = load_tokenizer_for_checkpoint(checkpoint_path)
        print(f"Resuming with tokenizer loaded from {checkpoint_path.parent / 'tokenizer.json'}")
    else:
        print(f"Training tokenizer on {len(corpus):,} characters...")
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=model_config.vocab_size)
        tokenizer.add_eos_token()
    # The model's vocab_size must include the reserved EOS row -- it can
    # only be added at pretraining time, before the embedding table is
    # built. See docs/eos-generation-stopping.md.
    model_config.vocab_size = tokenizer.vocab_size

    processed_path = Path(data_config["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    tokens_path = processed_path / "tokens.bin"
    meta_path = processed_path / "tokens.meta.json"
    # Reuses tokens.bin if it already holds the exact result of encoding
    # this corpus with this tokenizer -- see docs/token-cache.md. Falls
    # back to encoding (and rewriting the cache) whenever the corpus,
    # tokenizer, or tokens.bin itself has changed since the cache was
    # written; never silently reuses stale ids.
    cache_reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    print(
        "Reusing cached tokens.bin" if cache_reused else f"Encoded {len(corpus):,} characters"
    )

    all_tokens = load_token_ids(tokens_path)
    train_tokens, val_tokens = train_val_split(all_tokens, data_config["train_split"])
    train_dataset = TokenDataset(train_tokens, context_length=model_config.context_length)
    val_dataset = TokenDataset(val_tokens, context_length=model_config.context_length)

    model = GPT(model_config)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    start_step = 0
    optimizer_state_dict = None
    if args.resume:
        checkpoint = load_checkpoint_for_resume(train_config.checkpoint_dir, model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        start_step = checkpoint["step"]
        print(f"Resuming from step {start_step}/{train_config.max_steps}")

    result = train_model(
        model,
        train_dataset,
        val_dataset,
        train_config,
        device=device,
        tokenizer=tokenizer,
        start_step=start_step,
        optimizer_state_dict=optimizer_state_dict,
    )
    print(f"Checkpoint saved to {result['checkpoint_path']}")
    print(f"Tokenizer saved to {result['tokenizer_path']}")


if __name__ == "__main__":
    main()
