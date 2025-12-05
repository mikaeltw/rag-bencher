import pytest

from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.pipelines import multi_query

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_multi_query_builds_and_runs_offline() -> None:
    docs = load_texts_as_documents(["examples/data/sample.txt"])
    chain, debug = multi_query.build_chain(docs, model="dummy", k=2, n_queries=2)
    out = chain.invoke("What is LangChain?")
    assert isinstance(out, str) and len(out) > 0
    dbg = debug()
    assert dbg.get("pipeline") == "multi_query"
    assert len(dbg.get("retrieved", [])) >= 1


def test_dedupe_queries_limits_and_uniqueness() -> None:
    merged = multi_query._dedupe_queries("base", ["base", "extra", "extra", "third"], 2)
    assert merged == ["base", "extra"]
    merged_many = multi_query._dedupe_queries("base", ["one", "two", "three"], 5)
    assert merged_many[:3] == ["base", "one", "two"]


def test_fallback_queries_truncates() -> None:
    queries = multi_query._fallback_queries("question", 3)
    assert len(queries) == 3
    assert queries[0] == "question"


def test_dedupe_queries_respects_minimum_limit() -> None:
    result = multi_query._dedupe_queries("prompt", ["extra1", "extra2"], 0)
    # Limit zero still returns at least one query (the base prompt)
    assert result == ["prompt"]


def test_dedupe_queries_preserves_order() -> None:
    generated = ["gamma", "alpha", "gamma", "beta"]
    result = multi_query._dedupe_queries("alpha", generated, 4)
    assert result == ["alpha", "gamma", "beta"]
