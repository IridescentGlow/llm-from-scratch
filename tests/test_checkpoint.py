"""Tests for the shared, safe checkpoint loader. See docs/checkpoint-format.md."""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from llm_from_scratch.checkpoint import load_checkpoint_dict
from llm_from_scratch.model import GPT, GPTConfig

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _tiny_model_config(**overrides) -> GPTConfig:
    defaults = dict(vocab_size=20, context_length=8, n_layer=1, n_head=1, n_embd=8, dropout=0.0)
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _save_safe_checkpoint(path: Path, model_config: GPTConfig, **extra) -> None:
    torch.save(
        {
            "model_state_dict": GPT(model_config).state_dict(),
            "model_config": asdict(model_config),
            **extra,
        },
        path,
    )


def test_load_checkpoint_dict_loads_with_weights_only_true(tmp_path):
    """The whole point of this milestone: a checkpoint saved by the current
    code must load under weights_only=True -- proven directly here, not
    just indirectly through load_checkpoint_dict's own (also
    weights_only=True) call, by re-reading the raw file the same way.
    """
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / "latest.pt"
    _save_safe_checkpoint(checkpoint_path, model_config, optimizer_state_dict={}, step=3)

    raw = torch.load(checkpoint_path, weights_only=True)
    assert raw["model_config"] == asdict(model_config)


def test_load_checkpoint_dict_reconstructs_gptconfig(tmp_path):
    model_config = _tiny_model_config(n_embd=32, n_head=4)
    checkpoint_path = tmp_path / "latest.pt"
    _save_safe_checkpoint(checkpoint_path, model_config, optimizer_state_dict={}, step=1)

    checkpoint = load_checkpoint_dict(checkpoint_path)

    assert isinstance(checkpoint["model_config"], GPTConfig)
    assert checkpoint["model_config"] == model_config


def test_load_checkpoint_dict_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint_dict(tmp_path / "does_not_exist.pt")


def test_load_checkpoint_dict_rejects_legacy_pickled_config(tmp_path):
    """A checkpoint saved before this milestone has model_config pickled as
    a real GPTConfig object, not a plain dict -- weights_only=True refuses
    to unpickle it, and load_checkpoint_dict must turn that refusal into a
    clear, actionable error, never fall back to weights_only=False.
    """
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": GPT(model_config).state_dict(),
            "model_config": model_config,  # a real object, not asdict(...)
            "optimizer_state_dict": {},
            "step": 1,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="[Rr]egenerate"):
        load_checkpoint_dict(checkpoint_path)


def test_load_checkpoint_dict_never_falls_back_to_weights_only_false(tmp_path, monkeypatch):
    """Even if a legacy checkpoint is encountered, load_checkpoint_dict must
    not retry with weights_only=False -- confirmed by making a
    weights_only=False call fail loudly if it's ever attempted.
    """
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {"model_state_dict": GPT(model_config).state_dict(), "model_config": model_config},
        checkpoint_path,
    )

    import llm_from_scratch.checkpoint as checkpoint_module

    orig_load = checkpoint_module.torch.load

    def _guarded_load(*args, **kwargs):
        if kwargs.get("weights_only") is False:
            raise AssertionError("load_checkpoint_dict must never use weights_only=False")
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", _guarded_load)

    with pytest.raises(ValueError, match="[Rr]egenerate"):
        load_checkpoint_dict(checkpoint_path)


def test_no_source_file_calls_torch_load_with_weights_only_false():
    """Grep-level guardrail: nothing in src/ or scripts/ should ever call
    torch.load(..., weights_only=False) again -- load_checkpoint_dict is
    meant to be the one place checkpoints get loaded from. (Legacy-format
    detection relies specifically on weights_only=True's refusal -- a
    weights_only=False call anywhere would silently defeat that.)
    """
    pattern = re.compile(r"weights_only\s*=\s*False")
    offenders = []
    for directory in (SRC_DIR, SCRIPTS_DIR):
        for path in directory.rglob("*.py"):
            text = path.read_text()
            if pattern.search(text):
                offenders.append(str(path))
    assert offenders == []
