from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_bencher import cli

pytestmark = [pytest.mark.unit, pytest.mark.offline]

CLI_MOD = cast(Any, cli)


class DummyChain:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, question: str, config: Dict[str, Any] | None = None) -> str:
        self.calls.append({"question": question, "config": config or {}})
        return f"ans:{question}"


def _make_cfg() -> Any:
    model = SimpleNamespace(name="demo-model")
    retriever = SimpleNamespace(k=2)
    runtime = SimpleNamespace(offline=True, device="cpu")
    data = SimpleNamespace(paths=["doc.txt"])

    class DummyCfg:
        def __init__(self) -> None:
            self.model = model
            self.retriever = retriever
            self.runtime = runtime
            self.data = data
            self.provider = None

        def model_dump(self) -> Dict[str, Any]:
            return {"model": {"name": self.model.name}}

    return DummyCfg()


class CacheLog:
    def __init__(self) -> None:
        self.gets: list[tuple[str, str]] = []
        self.sets: list[tuple[str, str, str]] = []


def _patch_common(monkeypatch: pytest.MonkeyPatch, cfg: Any, docs: List[Document], chain: DummyChain) -> CacheLog:
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "load_texts_as_documents", lambda _: docs)
    monkeypatch.setattr(cli, "_pick_llm", lambda _cfg: "llm-object")
    monkeypatch.setattr(cli, "build_embeddings_adapter", lambda _cfg: None)
    monkeypatch.setattr(cli, "build_vector_backend", lambda _cfg: None)
    monkeypatch.setattr(
        CLI_MOD.naive_rag,
        "build_chain",
        lambda *args, **kwargs: (chain, lambda: {"pipeline": "naive"}),
    )
    log = CacheLog()

    def fake_get(model: str, prompt: str) -> str | None:
        log.gets.append((model, prompt))
        return None

    def fake_set(model: str, prompt: str, value: str) -> None:
        log.sets.append((model, prompt, value))

    monkeypatch.setattr(cli, "cache_get", fake_get)
    monkeypatch.setattr(cli, "cache_set", fake_set)
    return log


def test_cli_main_runs_chain_on_cache_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    docs = [Document(page_content="doc", metadata={"source": "doc.txt"})]
    chain = DummyChain()
    cache_log = _patch_common(monkeypatch, cfg, docs, chain)
    monkeypatch.setattr(sys, "argv", ["rag-bencher", "--config", "cfg.yaml", "--question", "What is RAG?"])

    cli.main()

    assert cache_log.gets == [("demo-model", "What is RAG?")]
    assert len(cache_log.sets) == 1
    assert chain.calls and chain.calls[0]["question"] == "What is RAG?"


def test_cli_main_leaves_device_env_when_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.device = "auto"
    docs = [Document(page_content="doc", metadata={"source": "doc.txt"})]
    chain = DummyChain()
    _patch_common(monkeypatch, cfg, docs, chain)
    monkeypatch.setattr(sys, "argv", ["rag-bencher", "--config", "cfg.yaml", "--question", "Auto?"])
    monkeypatch.delenv("RAG_BENCH_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cli.main()

    assert os.environ.get("RAG_BENCH_DEVICE") is None
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None


def test_cli_main_returns_cached_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    docs = [Document(page_content="doc", metadata={"source": "doc.txt"})]
    chain = DummyChain()
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "load_texts_as_documents", lambda _: docs)
    monkeypatch.setattr(cli, "_pick_llm", lambda _cfg: "llm-object")
    monkeypatch.setattr(cli, "build_embeddings_adapter", lambda _cfg: None)
    monkeypatch.setattr(cli, "build_vector_backend", lambda _cfg: None)
    monkeypatch.setattr(
        CLI_MOD.naive_rag,
        "build_chain",
        lambda *args, **kwargs: (chain, lambda: {"pipeline": "naive"}),
    )

    monkeypatch.setattr(sys, "argv", ["rag-bencher", "--config", "cfg.yaml", "--question", "Cached Q?"])

    def fake_get(model: str, prompt: str) -> str:
        return "cached-answer"

    monkeypatch.setattr(cli, "cache_get", fake_get)
    writes: list[tuple[str, str, str]] = []

    def record_cache(model: str, prompt: str, value: str) -> None:
        writes.append((model, prompt, value))

    monkeypatch.setattr(cli, "cache_set", record_cache)

    cli.main()

    assert not chain.calls
    assert writes == []


def test_cli_main_skips_embeddings_when_adapter_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.provider = SimpleNamespace(name="aws")
    cfg.model_dump = lambda: {"model": {"name": cfg.model.name}, "provider": {"name": "aws"}}
    docs = [Document(page_content="doc", metadata={"source": "doc.txt"})]
    chain = DummyChain()
    cache_log = _patch_common(monkeypatch, cfg, docs, chain)
    # Override the embeddings adapter builder to ensure it's invoked but returns None.
    calls: list[dict[str, Any]] = []

    def build_adapter(cfg_data: Dict[str, Any]) -> None:
        calls.append(cfg_data)
        return None

    monkeypatch.setattr(cli, "build_embeddings_adapter", build_adapter)
    monkeypatch.setattr(sys, "argv", ["rag-bencher", "--config", "cfg.yaml", "--question", "No embeddings?"])

    cli.main()

    assert calls and calls[0]["name"] == "aws"
    assert cache_log.gets == [("demo-model", "No embeddings?")]


def test_pick_llm_offline_builds_hf_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = True
    calls: dict[str, Any] = {}

    class DummyTokenizer:
        pad_token_id = None
        eos_token_id = 42

        @classmethod
        def from_pretrained(cls, model_id: str) -> "DummyTokenizer":
            calls["tokenizer_model"] = model_id
            return cls()

    class DummyGenerationConfig:
        def __init__(self) -> None:
            self.params: dict[str, Any] = {}

        def update(self, **kwargs: Any) -> None:
            self.params.update(kwargs)

    class DummyModel:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.generation_config = DummyGenerationConfig()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> DummyModel:
            calls["model_id"] = model_id
            return DummyModel(model_id)

    class DummyPipeline:
        __slots__ = ("model",)

        def __init__(self, model: DummyModel) -> None:
            self.model = model

    def fake_pipeline(**kwargs: Any) -> DummyPipeline:
        calls["pipeline_kwargs"] = kwargs
        return DummyPipeline(kwargs["model"])

    class DummyHFPipeline:
        def __init__(self, pipeline: DummyPipeline) -> None:
            self.pipeline = pipeline

        def bind(self, stop: List[str]) -> Any:
            return SimpleNamespace(bound_stop=stop, pipeline=self.pipeline)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSeq2SeqLM=AutoModel,
            AutoTokenizer=DummyTokenizer,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFacePipeline=DummyHFPipeline))

    llm = cli._pick_llm(cfg)
    llm_obj = cast(Any, llm)
    assert llm_obj.bound_stop == ["\nQuestion:", "###END"]
    assert calls["tokenizer_model"] == "google/flan-t5-small"
    assert calls["model_id"] == "google/flan-t5-small"


def test_pick_llm_offline_preserves_existing_pad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = True
    tracker: dict[str, Any] = {}

    class DummyTokenizer:
        pad_token_id = 99
        eos_token_id = 7

        @classmethod
        def from_pretrained(cls, model_id: str) -> "DummyTokenizer":
            tracker["tokenizer_model"] = model_id
            return cls()

    class DummyGenerationConfig:
        def __init__(self) -> None:
            self.params: dict[str, Any] = {}

        def update(self, **kwargs: Any) -> None:
            self.params.update(kwargs)

    class DummyModel:
        def __init__(self) -> None:
            self.generation_config = DummyGenerationConfig()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> DummyModel:
            tracker["model_id"] = model_id
            return DummyModel()

    class DummyPipeline:
        __slots__ = ("model", "tokenizer")

        def __init__(self, model: DummyModel, tokenizer: DummyTokenizer) -> None:
            self.model = model
            self.tokenizer = tokenizer

    def fake_pipeline(**kwargs: Any) -> DummyPipeline:
        tracker["pipeline_kwargs"] = kwargs
        return DummyPipeline(kwargs["model"], kwargs["tokenizer"])

    class DummyHFPipeline:
        def __init__(self, pipeline: DummyPipeline) -> None:
            self.pipeline = pipeline

        def bind(self, stop: List[str]) -> Any:
            return SimpleNamespace(bound_stop=stop, pipeline=self.pipeline)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSeq2SeqLM=AutoModel,
            AutoTokenizer=DummyTokenizer,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFacePipeline=DummyHFPipeline))

    llm = cli._pick_llm(cfg)
    llm_obj = cast(Any, llm)
    assert llm_obj.bound_stop == ["\nQuestion:", "###END"]
    # Existing pad token should remain untouched because it was already set.
    assert llm_obj.pipeline.tokenizer.pad_token_id == 99
    assert llm_obj.pipeline.tokenizer.eos_token_id == 7


def test_pick_llm_offline_handles_missing_generation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = True

    class DummyTokenizer:
        pad_token_id = None
        eos_token_id = 5

        @classmethod
        def from_pretrained(cls, model_id: str) -> "DummyTokenizer":
            return cls()

    class DummyModel:
        def __init__(self) -> None:
            # Intentionally no generation_config attribute
            self.model_id = "dummy"

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> DummyModel:
            return DummyModel()

    class DummyPipeline:
        __slots__ = ("model",)

        def __init__(self, model: DummyModel) -> None:
            self.model = model

    def fake_pipeline(**kwargs: Any) -> DummyPipeline:
        return DummyPipeline(kwargs["model"])

    class DummyHFPipeline:
        def __init__(self, pipeline: DummyPipeline) -> None:
            self.pipeline = pipeline

        def bind(self, stop: List[str]) -> Any:
            return SimpleNamespace(bound_stop=stop, pipeline=self.pipeline)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSeq2SeqLM=AutoModel,
            AutoTokenizer=DummyTokenizer,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFacePipeline=DummyHFPipeline))

    llm = cli._pick_llm(cfg)
    llm_obj = cast(Any, llm)
    assert llm_obj.bound_stop == ["\nQuestion:", "###END"]
    # Even without a generation_config attribute we should still build the pipeline.
    assert isinstance(llm_obj.pipeline, DummyPipeline)


def test_pick_llm_uses_provider_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = False
    cfg.provider = SimpleNamespace(name="aws")
    cfg.model_dump = lambda: {"provider": {"name": "aws"}}

    class DummyAdapter:
        def to_langchain(self) -> RunnableLambda[Any, str]:
            return RunnableLambda(lambda prompt: f"adapter:{prompt}")

    monkeypatch.setattr(cli, "build_chat_adapter", lambda _cfg: DummyAdapter())
    llm = cli._pick_llm(cfg)
    assert llm.invoke("hello") == "adapter:hello"


def test_pick_llm_errors_when_adapter_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = False
    cfg.provider = SimpleNamespace(name="azure")
    cfg.model_dump = lambda: {"provider": {"name": "azure"}, "model": {"name": cfg.model.name}}
    monkeypatch.setattr(cli, "build_chat_adapter", lambda _cfg: None)
    with pytest.raises(RuntimeError):
        cli._pick_llm(cfg)


def test_pick_llm_defaults_to_openai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.offline = False

    class DummyChat:
        __slots__ = ("kwargs",)

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=DummyChat))
    llm = cli._pick_llm(cfg)
    assert isinstance(llm, DummyChat)
    assert llm.kwargs == {"model": cfg.model.name, "temperature": 0}


def test_cli_main_sets_cuda_device_and_handles_vector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg()
    cfg.runtime.device = "cuda"
    cfg.runtime.offline = False
    cfg.provider = SimpleNamespace(name="aws")
    cfg.model_dump = lambda: {
        "model": {"name": cfg.model.name},
        "provider": {"name": "aws"},
        "vector": {"backend": "stub"},
    }

    docs = [Document(page_content="doc", metadata={"source": "doc.txt"})]
    chain = DummyChain()
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "load_texts_as_documents", lambda _: docs)
    monkeypatch.setattr(cli, "naive_rag", SimpleNamespace(build_chain=lambda *args, **kwargs: (chain, lambda: {})))
    monkeypatch.setattr(cli, "_pick_llm", lambda _cfg: "llm-object")

    class DummyAdapter:
        def to_langchain(self) -> str:
            return "embedding"

    monkeypatch.setattr(cli, "build_embeddings_adapter", lambda _cfg: DummyAdapter())

    class DummyVectorBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def make_retriever(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)
            raise RuntimeError("boom")

    vec_backend = DummyVectorBackend()
    monkeypatch.setattr(cli, "build_vector_backend", lambda _cfg: vec_backend)

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(cli, "cache_get", lambda *_args: None)
    monkeypatch.setattr(cli, "cache_set", lambda m, q, a: calls.append((m, q, a)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-bencher",
            "--config",
            "cfg.yaml",
            "--question",
            "Vector?",
        ],
    )

    monkeypatch.delenv("RAG_BENCH_DEVICE", raising=False)
    cli.main()

    assert os.environ.get("RAG_BENCH_DEVICE") == "cuda"
    assert vec_backend.calls, "vector backend should be invoked even when it fails"
    assert calls and calls[0][1] == "Vector?"
