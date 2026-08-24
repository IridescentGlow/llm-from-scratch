"""Tests for scripts/train.py's --resume tokenizer handling.

See docs/resume-tokenizer-consistency.md: --resume must load the tokenizer
saved next to the checkpoint being resumed, not retrain BPE on whatever
data/raw/ currently contains.

scripts/ has no __init__.py (scripts are thin entry points, not a package
-- see CLAUDE.md), so the script is loaded directly from its file path
rather than imported normally.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from llm_from_scratch.data import load_token_ids
from llm_from_scratch.tokenizer import BPETokenizer

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

CORPUS = "the quick brown fox jumps over the lazy dog. " * 60

CONFIG_TEMPLATE = """
model:
  vocab_size: 260
  context_length: 8
  n_layer: 1
  n_head: 1
  n_embd: 8
  dropout: 0.0
train:
  batch_size: 2
  learning_rate: 1e-2
  warmup_steps: 1
  max_steps: {max_steps}
  grad_clip: 1.0
  eval_every: {max_steps}
  checkpoint_dir: {checkpoint_dir}
  checkpoint_every: {checkpoint_every}
data:
  raw_path: {raw_dir}
  processed_path: {processed_dir}
  train_split: 0.8
"""


def _load_train_script():
    """Load scripts/train.py as a module, isolated per call."""
    spec = importlib.util.spec_from_file_location("scripts_train_under_test", SCRIPTS_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path, corpus: str = CORPUS):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "corpus.txt").write_text(corpus)
    processed_dir = tmp_path / "processed"
    checkpoint_dir = tmp_path / "ckpt"
    return raw_dir, processed_dir, checkpoint_dir


def _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir, **overrides) -> Path:
    params = dict(max_steps=4, checkpoint_every=4)
    params.update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            raw_dir=raw_dir, processed_dir=processed_dir, checkpoint_dir=checkpoint_dir, **params
        )
    )
    return config_path


def _run_main(monkeypatch, config_path, resume: bool = False) -> None:
    module = _load_train_script()
    argv = ["train.py", "--config", str(config_path), "--device", "cpu"]
    if resume:
        argv.append("--resume")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_fresh_training_creates_a_tokenizer(tmp_path, monkeypatch):
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir)

    _run_main(monkeypatch, config_path)

    tokenizer_path = checkpoint_dir / "tokenizer.json"
    assert tokenizer_path.exists()
    tokenizer = BPETokenizer.load(tokenizer_path)
    assert tokenizer.eos_token_id is not None
    assert len(tokenizer.merges) > 0


def test_resume_loads_the_saved_tokenizer_not_a_retrained_one(tmp_path, monkeypatch):
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=4, checkpoint_every=4
    )
    _run_main(monkeypatch, config_path)

    original_tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")

    # Changing the raw corpus after the checkpoint was made must not affect
    # the tokenizer resume uses -- a differently-worded corpus would train
    # different merges if retrained, per docs/resume-tokenizer-consistency.md.
    (raw_dir / "corpus.txt").write_text(
        "an entirely different sentence about something else altogether. " * 60
    )

    resume_config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=8, checkpoint_every=8
    )
    _run_main(monkeypatch, resume_config_path, resume=True)

    resumed_tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")
    assert resumed_tokenizer.merges == original_tokenizer.merges
    assert resumed_tokenizer.vocab == original_tokenizer.vocab
    assert resumed_tokenizer.eos_token_id == original_tokenizer.eos_token_id


def test_changing_raw_corpus_does_not_affect_resumed_training_data(tmp_path, monkeypatch):
    """Beyond just the saved tokenizer.json matching: encoding the (changed)
    corpus during resume must use the original tokenizer's ids, not ids from
    a tokenizer retrained on the new text.
    """
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir)
    _run_main(monkeypatch, config_path)
    original_tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")

    (raw_dir / "corpus.txt").write_text("completely different text here. " * 60)
    resume_config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=8, checkpoint_every=8
    )
    _run_main(monkeypatch, resume_config_path, resume=True)

    tokens_path = processed_dir / "tokens.bin"
    assert tokens_path.exists(), "resume should still write encoded tokens for the (new) corpus"
    written_ids = list(load_token_ids(tokens_path))
    # The encoded tokens must have come from the original (loaded) tokenizer's
    # merges applied to the *new* corpus text, not from a tokenizer retrained
    # on that new text.
    new_corpus_text = (raw_dir / "corpus.txt").read_text()
    assert written_ids == original_tokenizer.encode(new_corpus_text)


def test_resume_with_no_saved_tokenizer_fails_clearly(tmp_path, monkeypatch):
    """A checkpoint with no tokenizer.json (e.g. one that predates tokenizer
    persistence) must fail loudly on --resume, not silently retrain one.
    """
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    checkpoint_dir.mkdir()
    # A checkpoint file exists (so this isn't "no checkpoint at all"), but no
    # tokenizer.json sits next to it.
    torch.save(
        {"model_state_dict": {}, "model_config": None, "optimizer_state_dict": {}, "step": 1},
        checkpoint_dir / "latest.pt",
    )
    config_path = _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir)

    with pytest.raises(FileNotFoundError, match="tokenizer"):
        _run_main(monkeypatch, config_path, resume=True)


def test_resumed_training_still_reaches_expected_step(tmp_path, monkeypatch):
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=4, checkpoint_every=4
    )
    _run_main(monkeypatch, config_path)

    checkpoint = torch.load(checkpoint_dir / "latest.pt", weights_only=True)
    assert checkpoint["step"] == 4

    resume_config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=10, checkpoint_every=10
    )
    _run_main(monkeypatch, resume_config_path, resume=True)

    resumed_checkpoint = torch.load(checkpoint_dir / "latest.pt", weights_only=True)
    assert resumed_checkpoint["step"] == 10
