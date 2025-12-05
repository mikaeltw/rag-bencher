from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from rag_bencher.utils import factories, torch_utils

pytestmark = pytest.mark.gpu


def _patch_torch_module(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool = True) -> None:
    """Install a lightweight torch stub for the duration of the test."""

    class _CudaNS:
        def is_available(self) -> bool:
            return cuda_available

    stub = SimpleNamespace(cuda=_CudaNS())
    monkeypatch.setitem(sys.modules, "torch", stub)


@pytest.mark.gpu
def test_cuda_available_handles_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("torch missing in this environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert torch_utils.cuda_available() is False


@pytest.mark.gpu
def test_device_str_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: False)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)
    assert torch_utils.device_str() == "cuda"


@pytest.mark.gpu
def test_device_str_respects_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: True)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)
    assert torch_utils.device_str() == "cpu"


@pytest.mark.gpu
def test_to_device_uses_tensor_to_method(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch_module(monkeypatch, cuda_available=True)
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: False)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)

    class DummyTensor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def to(self, device: str) -> str:
            self.calls.append(device)
            return f"{device}-moved"

    tensor = DummyTensor()
    result = torch_utils.to_device(tensor)
    assert result == "cuda-moved"
    assert tensor.calls == ["cuda"]


@pytest.mark.gpu
def test_to_device_noop_without_to_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch_module(monkeypatch, cuda_available=True)
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: False)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)
    sentinel = object()
    assert torch_utils.to_device(sentinel) is sentinel


@pytest.mark.gpu
def test_new_tensor_passes_device_from_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_as_tensor(data: Any, dtype: Any = None, device: str | None = None) -> dict[str, Any]:
        captured["data"] = data
        captured["dtype"] = dtype
        captured["device"] = device
        return {"data": data, "dtype": dtype, "device": device}

    stub = SimpleNamespace(as_tensor=fake_as_tensor)
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setattr(torch_utils, "device_str", lambda: "cuda")

    torch_utils.new_tensor([1, 2, 3])
    assert captured == {"data": [1, 2, 3], "dtype": None, "device": "cuda"}


@pytest.mark.gpu
def test_preferred_device_favors_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: True)
    monkeypatch.setattr(factories, "cuda_available", lambda: True)
    assert factories._preferred_device() == "cpu"


@pytest.mark.gpu
def test_preferred_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)

    def fake_cuda_available() -> bool:
        return True

    monkeypatch.setattr(factories, "cuda_available", fake_cuda_available)
    assert factories._preferred_device() == "cuda"


@pytest.mark.gpu
def test_preferred_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: False)
    assert factories._preferred_device() == "cpu"
