"""Tests for scripts/train.py's token-cache reuse on --resume.

See docs/token-cache.md: --resume must not re-encode the corpus when the
checkpoint's tokenizer and the corpus are both unchanged since the cache
was written.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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
  min_lr: 1e-3
  warmup_steps: 1
  max_steps: {max_steps}
  grad_clip: 1.0
  eval_every: {max_steps}
  checkpoint_dir: {checkpoint_dir}
  checkpoint_every: {checkpoint_every}
  weight_decay: 0.01
data:
  raw_path: {raw_dir}
  processed_path: {processed_dir}
  train_split: 0.8
"""


def _load_train_script():
    spec = importlib.util.spec_from_file_location("scripts_train_under_test_cache", SCRIPTS_DIR / "train.py")
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


def test_resume_reuses_cache_without_reencoding(tmp_path, monkeypatch):
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir)
    _run_main(monkeypatch, config_path)

    meta_path = processed_dir / "tokens.meta.json"
    assert meta_path.exists()
    tokens_path = processed_dir / "tokens.bin"
    tokens_before = tokens_path.read_bytes()

    def fail_if_called(self, text):
        raise AssertionError("encode() should not run on --resume with an unchanged cache")

    monkeypatch.setattr(BPETokenizer, "encode", fail_if_called)

    resume_config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=8, checkpoint_every=8
    )
    _run_main(monkeypatch, resume_config_path, resume=True)

    assert tokens_path.read_bytes() == tokens_before


def test_resume_reencodes_when_corpus_changed(tmp_path, monkeypatch):
    """The corpus changing between the crashed run and --resume must not be
    silently ignored -- the cache must be rejected and re-encoded."""
    raw_dir, processed_dir, checkpoint_dir = _setup(tmp_path)
    config_path = _write_config(tmp_path, raw_dir, processed_dir, checkpoint_dir)
    _run_main(monkeypatch, config_path)

    tokens_path = processed_dir / "tokens.bin"
    tokens_before = tokens_path.read_bytes()

    (raw_dir / "corpus.txt").write_text("an entirely different corpus text. " * 60)
    resume_config_path = _write_config(
        tmp_path, raw_dir, processed_dir, checkpoint_dir, max_steps=8, checkpoint_every=8
    )
    _run_main(monkeypatch, resume_config_path, resume=True)

    assert tokens_path.read_bytes() != tokens_before
