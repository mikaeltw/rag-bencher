from __future__ import annotations

from typing import Any, Dict, List, cast

import pytest
from langchain_core.documents import Document

from rag_bencher.config import load_config
from rag_bencher.pipelines import hyde, multi_query, naive_rag, rerank
from rag_bencher.pipelines.selector import PipelineSelection, select_pipeline


class DummyChain:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: List[str] = []

    def invoke(self, question: str) -> str:
        self.calls.append(question)
        return f"{self.tag}:{question}"


def make_stub_builder(tag: str, store: Dict[str, Any]) -> DummyChain:
    chain = DummyChain(tag)

    def builder(docs: List[Document], **kwargs: Any) -> tuple[DummyChain, Any]:
        store["docs"] = docs
        store["kwargs"] = kwargs
        return chain, (lambda: {"pipeline": tag})

    store["builder"] = builder
    return chain


@pytest.mark.unit
def test_select_pipeline_defaults_to_naive(monkeypatch: pytest.MonkeyPatch) -> None:
    bench_cfg = load_config("configs/wiki.yaml")
    store: Dict[str, Any] = {}
    chain = make_stub_builder("naive", store)
    monkeypatch.setattr(naive_rag, "build_chain", store["builder"])
    monkeypatch.setattr("rag_bencher.pipelines.selector.load_config", lambda *_: pytest.fail("load_config reused"))

    selection = select_pipeline("configs/wiki.yaml", docs=[Document(page_content="a")], cfg=bench_cfg)

    assert isinstance(selection, PipelineSelection)
    assert selection.config is bench_cfg
    assert selection.pipeline_id == "naive"
    selected_chain = cast(DummyChain, selection.chain)
    assert selected_chain is chain
    assert selection.debug() == {"pipeline": "naive"}
    assert store["kwargs"]["model"] == bench_cfg.model.name


@pytest.mark.unit
def test_select_pipeline_multi_query_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, Any] = {}
    _ = make_stub_builder("multi_query", store)
    monkeypatch.setattr(multi_query, "build_chain", store["builder"])

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unexpected pipeline invoked")

    monkeypatch.setattr(naive_rag, "build_chain", fail)
    monkeypatch.setattr(hyde, "build_chain", fail)
    monkeypatch.setattr(rerank, "build_chain", fail)

    selection = select_pipeline("configs/multi_query.yaml", docs=[])

    assert selection.pipeline_id == "multi_query"
    selection.chain.invoke("Q?")
    assert store["kwargs"]["n_queries"] == 3
    assert selection.debug() == {"pipeline": "multi_query"}


@pytest.mark.unit
def test_select_pipeline_hyde_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, Any] = {}
    _ = make_stub_builder("hyde", store)
    monkeypatch.setattr(hyde, "build_chain", store["builder"])

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unexpected pipeline invoked")

    monkeypatch.setattr(naive_rag, "build_chain", fail)
    monkeypatch.setattr(multi_query, "build_chain", fail)
    monkeypatch.setattr(rerank, "build_chain", fail)

    selection = select_pipeline("configs/hyde.yaml", docs=[])

    assert selection.pipeline_id == "hyde"
    selection.chain.invoke("Explain HYDE")
    assert selection.debug() == {"pipeline": "hyde"}


@pytest.mark.unit
def test_select_pipeline_rerank_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, Any] = {}
    _ = make_stub_builder("rerank", store)
    monkeypatch.setattr(rerank, "build_chain", store["builder"])

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unexpected pipeline invoked")

    monkeypatch.setattr(naive_rag, "build_chain", fail)
    monkeypatch.setattr(multi_query, "build_chain", fail)
    monkeypatch.setattr(hyde, "build_chain", fail)

    selection = select_pipeline("configs/rerank.yaml", docs=[])

    assert selection.pipeline_id == "rerank"
    selection.chain.invoke("Rank these")
    assert store["kwargs"]["rerank_top_k"] == 4
    assert store["kwargs"]["method"] == "cosine"
    assert selection.debug() == {"pipeline": "rerank"}


@pytest.mark.unit
def test_select_pipeline_builds_provider_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, Any] = {}
    chain = make_stub_builder("naive", store)
    monkeypatch.setattr(naive_rag, "build_chain", store["builder"])

    chat_calls: Dict[str, Any] = {}
    emb_calls: Dict[str, Any] = {}

    class DummyAdapter:
        def __init__(self, tag: str, calls: Dict[str, Any]) -> None:
            self._tag = tag
            self._calls = calls

        def to_langchain(self) -> str:
            result = f"{self._tag}-adapter"
            self._calls["returned"] = result
            return result

    def fake_chat(cfg: Dict[str, Any] | None) -> DummyAdapter:
        chat_calls["cfg"] = cfg
        return DummyAdapter("chat", chat_calls)

    def fake_emb(cfg: Dict[str, Any] | None) -> DummyAdapter:
        emb_calls["cfg"] = cfg
        return DummyAdapter("emb", emb_calls)

    monkeypatch.setattr("rag_bencher.pipelines.selector.build_chat_adapter", fake_chat)
    monkeypatch.setattr("rag_bencher.pipelines.selector.build_embeddings_adapter", fake_emb)

    selection = select_pipeline("configs/providers/aws.yaml", docs=[])

    assert selection.pipeline_id == "naive"
    assert chat_calls["cfg"]["name"] == "aws"
    assert emb_calls["cfg"]["name"] == "aws"
    assert store["kwargs"]["llm"] == "chat-adapter"
    assert store["kwargs"]["embeddings"] == "emb-adapter"
    selected_chain = cast(DummyChain, selection.chain)
    assert selected_chain is chain
    assert selection.debug() == {"pipeline": "naive"}
