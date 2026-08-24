"""
Entry point for text generation from a checkpoint.
Usage: python scripts/generate.py --checkpoint checkpoints/latest.pt \
    --prompt "..." [--max-new-tokens 50] [--temperature 0.0]
"""
import argparse

from llm_from_scratch.device import resolve_device
from llm_from_scratch.finetune import load_pretrained_model, load_tokenizer_for_checkpoint
from llm_from_scratch.generate import generate
from llm_from_scratch.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for reproducible sampling when --temperature > 0. Omit for "
            "today's unseeded behavior. See docs/reproducibility.md."
        ),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.seed is not None:
        set_seed(args.seed)

    model = load_pretrained_model(args.checkpoint)

    # Loads the exact tokenizer train_model saved alongside this checkpoint
    # -- see docs/01-tokenization.md, "Tokenizer persistence". No corpus or
    # --config needed anymore: the tokenizer no longer has to be retrained.
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
