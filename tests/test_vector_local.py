from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any, Sequence, cast

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rag_bencher.vector import local

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def _clear_caches() -> None:
    local._resolve_factory.cache_clear()
    local._faiss_safe_to_import.cache_clear()


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def test_build_local_vectorstore_uses_resolved_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()

    class DummyStore:
        calls: dict[str, Any] = {}
        _vector_store: VectorStore = cast(VectorStore, types.SimpleNamespace(name="vector-store"))

        @classmethod
        def from_documents(cls, docs: Sequence[Document], embeddings: Embeddings) -> VectorStore:
            cls.calls["args"] = (docs, embeddings)
            return cls._vector_store

    monkeypatch.setattr(local, "_resolve_factory", lambda: DummyStore)
    docs = [Document(page_content="one"), Document(page_content="two")]
    embeddings = DummyEmbeddings()
    result = local.build_local_vectorstore(docs, embeddings)
    assert result is DummyStore._vector_store
    call_docs, call_emb = DummyStore.calls["args"]
    assert [doc.page_content for doc in call_docs] == ["one", "two"]
    assert call_emb is embeddings


def test_resolve_factory_prefers_inmemory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setenv("RAG_BENCH_VECTORSTORE", "memory")
    sentinel = object()
    monkeypatch.setattr(local, "_inmemory_factory", lambda: sentinel)
    assert local._resolve_factory() is sentinel


def test_resolve_factory_disables_faiss_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.delenv("RAG_BENCH_VECTORSTORE", raising=False)
    monkeypatch.setenv("RAG_BENCH_DISABLE_FAISS", "true")
    sentinel = object()
    monkeypatch.setattr(local, "_inmemory_factory", lambda: sentinel)
    assert local._resolve_factory() is sentinel


def test_resolve_factory_allows_faiss_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setenv("RAG_BENCH_VECTORSTORE", "faiss")
    monkeypatch.setenv("RAG_BENCH_DISABLE_FAISS", "false")
    monkeypatch.setattr(local, "_faiss_safe_to_import", lambda: True)
    sentinel = object()
    monkeypatch.setattr(local, "_faiss_factory", lambda: sentinel)
    assert local._resolve_factory() is sentinel


def test_resolve_factory_errors_when_faiss_requested_but_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setenv("RAG_BENCH_VECTORSTORE", "faiss")
    monkeypatch.setattr(local, "_faiss_safe_to_import", lambda: False)
    with pytest.raises(RuntimeError, match="FAISS is unavailable"):
        local._resolve_factory()


def test_resolve_factory_errors_on_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setenv("RAG_BENCH_VECTORSTORE", "weird")
    with pytest.raises(ValueError, match="Unknown RAG_BENCH_VECTORSTORE"):
        local._resolve_factory()


def test_faiss_safe_to_import_handles_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert local._faiss_safe_to_import() is False


def test_faiss_safe_to_import_handles_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: cast(ModuleSpec, object()))

    def raise_error(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "cmd")

    monkeypatch.setattr(subprocess, "run", raise_error)
    assert local._faiss_safe_to_import() is False


def test_faiss_safe_to_import_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_caches()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: cast(ModuleSpec, object()))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    assert local._faiss_safe_to_import() is True


def test_faiss_factory_imports_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFAISS(VectorStore):
        pass

    module = types.SimpleNamespace(FAISS=DummyFAISS)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores.faiss", module)

    assert local._faiss_factory() is DummyFAISS


@pytest.mark.parametrize(
    "value",
    [None, "", " 0 ", "no", "off"],
)
def test_is_truthy_falsey_values(value: str | None) -> None:
    assert local._is_truthy(value) is False


@pytest.mark.parametrize("value", ["1", "True", " YES ", "on"])
def test_is_truthy_true_values(value: str) -> None:
    assert local._is_truthy(value) is True
