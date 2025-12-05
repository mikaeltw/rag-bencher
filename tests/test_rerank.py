import pytest

from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.pipelines.rerank import build_chain

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_rerank_cosine_fallback_runs() -> None:
    docs = load_texts_as_documents(["examples/data/sample.txt"])
    chain, debug = build_chain(docs, model="dummy", k=6, rerank_top_k=3, method="cosine")
    out = chain.invoke("What is LangChain?")
    assert isinstance(out, str) and len(out) > 0
    dbg = debug()
    assert dbg.get("pipeline") == "rerank"
    assert dbg.get("method") == "cosine"
    assert len(dbg.get("candidates", [])) >= 1
