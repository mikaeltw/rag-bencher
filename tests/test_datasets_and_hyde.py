import pytest

from rag_bench.eval.datasets import load_dataset
from rag_bench.pipelines.hyde import build_chain

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_dataset_loading_and_hyde_chain() -> None:
    docs = load_dataset("docs/wiki")
    chain, debug = build_chain(docs, model="dummy", k=2)
    out = chain.invoke("What is LangChain?")
    assert isinstance(out, str) and len(out) > 0
    info = debug()
    assert info.get("pipeline") == "hyde"
