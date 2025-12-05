from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, List, cast

import pytest

from rag_bencher import cli

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_pick_llm_offline_updates_generation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = cast(
        Any,
        SimpleNamespace(
            runtime=SimpleNamespace(offline=True),
            model=SimpleNamespace(name="demo"),
        ),
    )

    class DummyTokenizer:
        pad_token_id = None
        eos_token_id = 7

        @classmethod
        def from_pretrained(cls, model_id: str) -> DummyTokenizer:
            assert model_id == "google/flan-t5-small"
            return cls()

    class DummyGenerationConfig:
        def __init__(self) -> None:
            self.updates: dict[str, Any] = {}

        def update(self, **kwargs: Any) -> None:
            self.updates.update(kwargs)

    class DummyModel:
        def __init__(self) -> None:
            self.generation_config = DummyGenerationConfig()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> DummyModel:
            assert model_id == "google/flan-t5-small"
            return DummyModel()

    class DummyPipeline:
        __slots__ = ("model",)

        def __init__(self, *, model: DummyModel) -> None:
            self.model = model

    class DummyHFPipeline:
        __slots__ = ("pipeline",)

        def __init__(self, pipeline: DummyPipeline) -> None:
            self.pipeline = pipeline

        def bind(self, stop: List[str]) -> SimpleNamespace:
            return SimpleNamespace(bound_stop=stop, pipeline=self.pipeline)

    def fake_pipeline(**kwargs: Any) -> DummyPipeline:
        return DummyPipeline(model=kwargs["model"])

    monkeypatch.setitem(
        os.environ,
        "RAG_BENCH_OFFLINE_MODEL",
        "google/flan-t5-small",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSeq2SeqLM=AutoModel,
            AutoTokenizer=DummyTokenizer,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFacePipeline=DummyHFPipeline),
    )

    llm = cli._pick_llm(cfg)
    llm_data = cast(Any, llm)
    assert llm_data.bound_stop == ["\nQuestion:", "###END"]
    generation_updates = llm_data.pipeline.model.generation_config.updates
    assert generation_updates["do_sample"] is False
