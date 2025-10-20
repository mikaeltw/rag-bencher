
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.pipelines.multi_query import build_chain

def test_multi_query_builds_and_runs_offline():
    docs = load_texts_as_documents(['examples/data/sample.txt'])
    chain, debug = build_chain(docs, model="dummy", k=2, n_queries=2)
    out = chain.invoke("What is LangChain?")
    assert isinstance(out, str) and len(out) > 0
    dbg = debug()
    assert dbg.get("pipeline") == "multi_query"
    assert len(dbg.get("retrieved", [])) >= 1
