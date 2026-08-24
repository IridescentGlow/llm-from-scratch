"""Loading a pretrained checkpoint to continue training from. See docs/06-finetuning.md."""

from __future__ import annotations

from pathlib import Path

from llm_from_scratch.checkpoint import load_checkpoint_dict
from llm_from_scratch.model import GPT
from llm_from_scratch.tokenizer import BPETokenizer


def load_pretrained_model(checkpoint_path: str | Path) -> GPT:
    """Rebuild a GPT from a Stage 4 checkpoint (model_config + model_state_dict).

    Always loads weights onto CPU first (`map_location="cpu"`), regardless
    of what device the checkpoint was saved from -- a checkpoint saved on a
    CUDA machine would otherwise fail to load at all on a CPU-only machine.
    See docs/device-support.md. Callers move the model to the actual target
    device afterward (`train_model`/`evaluate_model`/`generate` all do this
    via their `device` argument).

    Loads with `weights_only=True` (via `load_checkpoint_dict`) -- see
    docs/checkpoint-format.md. Raises ValueError, not a silent unsafe
    fallback, for a checkpoint saved before that milestone.
    """
    checkpoint = load_checkpoint_dict(checkpoint_path, map_location="cpu")
    model = GPT(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_tokenizer_for_checkpoint(checkpoint_path: str | Path) -> BPETokenizer:
    """Load the tokenizer `train_model` saved next to `checkpoint_path`.

    See docs/01-tokenization.md, "Tokenizer persistence": a checkpoint's
    weights are only meaningful under the exact tokenizer that produced
    their training data, so this loads the saved tokenizer rather than
    retraining one from a corpus (retraining can silently produce a
    different id-to-string mapping, even at the same vocab_size).

    Raises FileNotFoundError, with an explicit explanation, for a checkpoint
    saved before tokenizer persistence existed -- there is no safe way to
    recover the original tokenizer for such a checkpoint, so this refuses
    rather than silently retraining a possibly-different one.
    """
    tokenizer_path = Path(checkpoint_path).parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"No tokenizer.json found next to {checkpoint_path}. "
            "This checkpoint predates tokenizer persistence, so its exact "
            "training-time tokenizer can't be recovered safely -- retrain it "
            "with the current scripts/train.py, which now saves "
            "tokenizer.json alongside the checkpoint."
        )
    return BPETokenizer.load(tokenizer_path)
