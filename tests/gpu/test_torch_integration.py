from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_bencher.utils import torch_utils


@pytest.mark.gpu
def test_cuda_available_with_stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_torch = cast(Any, types.ModuleType("torch"))
    dummy_torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    assert torch_utils.cuda_available() is True


@pytest.mark.gpu
def test_cuda_available_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("torch", None)
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("torch"):
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert torch_utils.cuda_available() is False


@pytest.mark.gpu
def test_device_str_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_utils, "wants_cpu", lambda: False)
    monkeypatch.setattr(torch_utils, "cuda_available", lambda: True)
    assert torch_utils.device_str() == "cuda"


@pytest.mark.gpu
def test_device_str_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:  # pragma: no cover - trivial helper
        raise RuntimeError("boom")

    monkeypatch.setattr(torch_utils, "wants_cpu", boom)
    assert torch_utils.device_str() == "cpu"


@pytest.mark.gpu
def test_to_device_moves_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_torch = cast(Any, types.ModuleType("torch"))
    dummy_torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(torch_utils, "device_str", lambda: "cuda")

    class TensorLike:
        def __init__(self) -> None:
            self.moved: list[str] = []

        def to(self, device: str) -> str:
            self.moved.append(device)
            return f"tensor@{device}"

    tensor = TensorLike()
    result = torch_utils.to_device(tensor)
    assert result == "tensor@cuda"
    assert tensor.moved == ["cuda"]


@pytest.mark.gpu
def test_to_device_handles_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("torch", None)
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("torch"):
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sentinel = object()
    assert torch_utils.to_device(sentinel) is sentinel


@pytest.mark.gpu
def test_new_tensor_uses_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTorch(types.ModuleType):
        __slots__ = ("calls",)

        def __init__(self) -> None:
            super().__init__("torch")
            self.calls: list[tuple[object, object | None, object | None]] = []
            self.cuda = SimpleNamespace(is_available=lambda: True)

        def as_tensor(
            self,
            data: object,
            *,
            dtype: object | None = None,
            device: object | None = None,
        ) -> tuple[object, object | None, object | None]:
            self.calls.append((data, dtype, device))
            return (data, dtype, device)

    dummy_torch = DummyTorch()
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(torch_utils, "device_str", lambda: "cuda")

    result = torch_utils.new_tensor([1, 2, 3], dtype=None)
    assert result == ([1, 2, 3], None, "cuda")
    assert dummy_torch.calls == [([1, 2, 3], None, "cuda")]
