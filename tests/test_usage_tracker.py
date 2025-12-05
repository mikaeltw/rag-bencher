from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.outputs import Generation, LLMResult

from rag_bencher.utils.callbacks.usage import UsageTracker


@pytest.mark.unit
def test_usage_tracker_counts_tokens() -> None:
    tracker = UsageTracker()
    tracker.on_llm_start({}, ["hello world", "second prompt"])

    response = LLMResult(generations=[[Generation(text="foo bar baz")]])
    tracker.on_llm_end(response)
    summary = tracker.summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 4  # hello world (2) + second prompt (2)
    assert summary["output_tokens"] == 3


@pytest.mark.unit
def test_usage_tracker_handles_malformed_response() -> None:
    tracker = UsageTracker()
    bad_response = cast(LLMResult, SimpleNamespace(generations=None))
    tracker.on_llm_end(bad_response)
    assert tracker.summary()["calls"] == 1
