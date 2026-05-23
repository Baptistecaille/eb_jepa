import torch

from eb_jepa.training_utils import resolve_amp_dtype


def test_resolve_amp_dtype_prefers_bfloat16_on_ampere_and_newer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    dtype = resolve_amp_dtype("auto", torch.device("cuda"))

    assert dtype == torch.bfloat16


def test_resolve_amp_dtype_falls_back_to_float16_on_older_cudas(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))

    dtype = resolve_amp_dtype("auto", torch.device("cuda"))

    assert dtype == torch.float16


def test_resolve_amp_dtype_honors_explicit_requests():
    assert resolve_amp_dtype("bf16", torch.device("cuda")) == torch.bfloat16
    assert resolve_amp_dtype("float16", torch.device("cuda")) == torch.float16
