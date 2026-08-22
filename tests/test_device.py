import pytest

from llm_from_scratch.device import resolve_device


def test_auto_detect_falls_back_to_cpu_when_nothing_else_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert resolve_device(None) == "cpu"


def test_auto_detect_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device(None) == "cuda"


def test_auto_detect_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device(None) == "mps"


def test_explicit_cpu_always_works():
    assert resolve_device("cpu") == "cpu"


def test_explicit_cuda_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


def test_explicit_mps_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        resolve_device("mps")


def test_explicit_cuda_succeeds_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_unknown_device_raises_value_error():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
