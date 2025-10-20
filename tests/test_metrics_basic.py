
from rag_bench.eval.metrics import lexical_f1, bow_cosine, context_recall

def test_metrics_sanity():
    a = "LangChain is a framework for LLM apps"
    b = "LangChain framework for language model applications"
    f1 = lexical_f1(a, b)
    cos = bow_cosine(a, b)
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= cos <= 1.0

    ctx = "This text mentions LangChain and language model apps."
    rec = context_recall(b, ctx)
    assert 0.0 <= rec <= 1.0
