from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_bencher.utils import factories, repro

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_preferred_device_respects_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: True)
    monkeypatch.setattr(factories, "cuda_available", lambda: True)
    assert factories._preferred_device() == "cpu"


def test_preferred_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: True)
    assert factories._preferred_device() == "cuda"


def test_preferred_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: False)
    assert factories._preferred_device() == "cpu"


def test_make_hf_embeddings_sets_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbeddings:
        __slots__ = ("model_name", "model_kwargs", "encode_kwargs")

        def __init__(self, *, model_name: str, model_kwargs: dict[str, Any], encode_kwargs: dict[str, Any]) -> None:
            self.model_name = model_name
            self.model_kwargs = model_kwargs
            self.encode_kwargs = encode_kwargs

    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=DummyEmbeddings))
    monkeypatch.setattr(factories, "wants_cpu", lambda: False)
    monkeypatch.setattr(factories, "cuda_available", lambda: False)

    emb = factories.make_hf_embeddings("mini", encode_kwargs={"normalize": True})
    assert isinstance(emb, DummyEmbeddings)
    assert emb.model_name == "mini"
    assert emb.model_kwargs["device"] == "cpu"
    assert emb.encode_kwargs == {"normalize": True}


def test_set_seeds_sets_numpy_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    dummy_np = cast(Any, types.ModuleType("numpy"))
    dummy_np.random = SimpleNamespace(seed=lambda value: calls.append(value))
    monkeypatch.setitem(sys.modules, "numpy", dummy_np)

    repro.set_seeds(77)
    assert calls == [77]


def test_set_seeds_handles_missing_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import: Any = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    repro.set_seeds(13)


def test_make_run_id_returns_hex() -> None:
    token = repro.make_run_id()
    assert len(token) == 10 and all(c in "0123456789abcdef" for c in token)
