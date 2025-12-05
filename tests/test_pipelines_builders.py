from __future__ import annotations

from typing import Any, List, cast

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from rag_bencher.pipelines import base as pipelines_base
from rag_bencher.pipelines import hyde, multi_query, naive_rag, rerank

pytestmark = [pytest.mark.unit, pytest.mark.offline]


class FakeEmbeddings:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_query(self, text: str) -> List[float]:
        self.seen.append(text)
        return [float(len(text) or 1.0), 1.0]


class FakeVectorStore:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.queries: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self.queries.append((query, k))
        return self.docs[:k]

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None) -> RunnableLambda[Any, list[Document]]:
        limit = (search_kwargs or {}).get("k", len(self.docs))
        return RunnableLambda(lambda _: self.docs[:limit])


class DummyRetriever(BaseRetriever):
    def __init__(self, docs: list[Document]) -> None:
        super().__init__()
        self._docs = docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        return self._docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        return self._docs


def _patch_common_builders(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySplitter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def split_documents(self, docs: list[Document]) -> list[Document]:
            return docs

    monkeypatch.setattr(module, "RecursiveCharacterTextSplitter", lambda *args, **kwargs: DummySplitter())
    monkeypatch.setattr(module, "make_hf_embeddings", lambda **kwargs: FakeEmbeddings())
    monkeypatch.setattr(module, "build_local_vectorstore", lambda docs, embed: FakeVectorStore(list(docs)))
    monkeypatch.setattr(
        module,
        "resolve_chat_llm",
        lambda *args, **kwargs: cast(RunnableSerializable[Any, Any], RunnableLambda(lambda text, **__: f"LLM:{text}")),
    )


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="alpha document", metadata={"source": "a"}),
        Document(page_content="beta information", metadata={"source": "b"}),
    ]


def test_hyde_chain_uses_fallback(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    _patch_common_builders(hyde, monkeypatch)
    monkeypatch.setattr(hyde, "has_openai_key", lambda: False)
    chain, debug = hyde.build_chain(docs, model="stub", k=1)
    answer = chain.invoke("What is alpha?")
    assert answer.startswith("LLM:")
    info = debug()
    assert info["pipeline"] == "hyde"
    assert info["retrieved"]
    assert "alpha" in info["hypothesis"]


def test_hyde_chain_can_generate_with_openai(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    _patch_common_builders(hyde, monkeypatch)
    monkeypatch.setattr(hyde, "has_openai_key", lambda: True)
    monkeypatch.setattr(hyde, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda prompt: f"gen::{prompt}"))

    chain, debug = hyde.build_chain(docs, model="stub", k=1)
    chain.invoke("Explain beta")
    assert debug()["hypothesis"].startswith("gen::")


def test_multi_query_chain_uses_fallback(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    _patch_common_builders(multi_query, monkeypatch)
    monkeypatch.setattr(multi_query, "has_openai_key", lambda: False)
    chain, debug = multi_query.build_chain(docs, model="stub", k=1, n_queries=2)
    ans = chain.invoke("Key facts?")
    assert ans.startswith("LLM:")
    info = debug()
    assert info["pipeline"] == "multi_query"
    assert len(info["queries"]) == 2


def test_multi_query_chain_deduplicates_generated_queries(
    monkeypatch: pytest.MonkeyPatch,
    docs: list[Document],
) -> None:
    _patch_common_builders(multi_query, monkeypatch)
    monkeypatch.setattr(multi_query, "has_openai_key", lambda: True)
    monkeypatch.setattr(
        multi_query,
        "ChatOpenAI",
        lambda **kwargs: RunnableLambda(lambda prompt: "facts\nfacts\nextra"),
    )

    chain, debug = multi_query.build_chain(docs, model="stub", k=1, n_queries=3)
    chain.invoke("alpha?")
    queries = debug()["queries"]
    assert queries[0] == "alpha?"
    assert len(queries) == 3


def test_naive_rag_chain_accepts_custom_retriever(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    monkeypatch.setattr(
        naive_rag,
        "resolve_chat_llm",
        lambda *args, **kwargs: cast(RunnableSerializable[Any, Any], RunnableLambda(lambda text, **__: f"LLM:{text}")),
    )
    retriever = DummyRetriever(docs)
    override_llm = cast(RunnableSerializable[Any, Any], RunnableLambda(lambda text, **__: text))
    chain, meta = naive_rag.build_chain(
        docs,
        retriever=retriever,
        llm=override_llm,
    )
    out = chain.invoke("Alpha?")
    assert "Question" in out
    assert meta() == {"pipeline": "naive_rag"}


def test_naive_rag_chain_builds_vector_store(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    _patch_common_builders(naive_rag, monkeypatch)
    chain, _ = naive_rag.build_chain(docs, k=1)
    result = chain.invoke("Beta?")
    assert result.startswith("LLM:")


def test_rerank_chain_produces_debug(monkeypatch: pytest.MonkeyPatch, docs: list[Document]) -> None:
    _patch_common_builders(rerank, monkeypatch)
    chain, debug = rerank.build_chain(docs, k=2, rerank_top_k=1)
    chain.invoke("alpha")
    info = debug()
    assert info["pipeline"] == "rerank"
    assert info["candidates"]


def test_cosine_handles_zero_vectors() -> None:
    assert rerank._cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert rerank._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_rag_pipeline_is_abstract() -> None:
    with pytest.raises(TypeError):
        cast(type[Any], pipelines_base.RagPipeline)()
