from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from rag_bencher.pipelines import utils

pytestmark = pytest.mark.unit


def test_has_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert utils.has_openai_key() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert utils.has_openai_key() is True


def test_resolve_chat_llm_prefers_override() -> None:
    override = cast(RunnableSerializable[Any, Any], RunnableLambda(lambda _: "override"))
    assert utils.resolve_chat_llm("demo", override=override) is override


def test_resolve_chat_llm_uses_openai_when_key(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    class DummyChat:
        def __init__(self, model: str, temperature: float) -> None:
            called["model"] = model
            called["temperature"] = temperature

    monkeypatch.setattr(utils, "has_openai_key", lambda: True)
    monkeypatch.setattr(utils, "ChatOpenAI", DummyChat)
    llm = utils.resolve_chat_llm("gpt-test")
    assert isinstance(llm, DummyChat)
    assert called == {"model": "gpt-test", "temperature": 0.0}


def test_resolve_chat_llm_offline_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "has_openai_key", lambda: False)
    llm = utils.resolve_chat_llm("demo")
    result = llm.invoke("Question?")
    assert "[offline answer]" in result
    assert "Question?" in result
