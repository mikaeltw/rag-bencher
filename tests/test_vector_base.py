from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from rag_bencher.vector import base

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def _install_backend_module(monkeypatch: pytest.MonkeyPatch, module_path: str, class_name: str) -> type[Any]:
    module = types.ModuleType(module_path)

    class DummyBackend:
        __slots__ = ("cfg",)

        def __init__(self, cfg: dict[str, Any]) -> None:
            self.cfg = cfg

    setattr(module, class_name, DummyBackend)
    monkeypatch.setitem(sys.modules, module_path, module)
    return DummyBackend


def test_build_vector_backend_returns_none_when_missing() -> None:
    assert base.build_vector_backend(None) is None
    assert base.build_vector_backend({}) is None


def test_build_vector_backend_selects_azure_ai_search(monkeypatch: pytest.MonkeyPatch) -> None:
    DummyBackend = _install_backend_module(monkeypatch, "rag_bencher.vector.azure_ai_search", "AzureAISearchBackend")
    backend = base.build_vector_backend({"name": "azure_ai_search", "url": "https://search"})
    assert isinstance(backend, DummyBackend)
    assert backend.cfg["url"] == "https://search"


def test_build_vector_backend_selects_opensearch(monkeypatch: pytest.MonkeyPatch) -> None:
    DummyBackend = _install_backend_module(monkeypatch, "rag_bencher.vector.opensearch", "OpenSearchBackend")
    backend = base.build_vector_backend({"name": "opensearch", "host": "localhost"})
    assert isinstance(backend, DummyBackend)
    assert backend.cfg["host"] == "localhost"


def test_build_vector_backend_selects_matching_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    DummyBackend = _install_backend_module(monkeypatch, "rag_bencher.vector.matching_engine", "MatchingEngineBackend")
    backend = base.build_vector_backend({"name": "matching_engine", "index": "demo"})
    assert isinstance(backend, DummyBackend)
    assert backend.cfg["index"] == "demo"


def test_build_vector_backend_errors_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown vector backend"):
        base.build_vector_backend({"name": "unsupported"})
