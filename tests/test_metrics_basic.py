import pytest

from rag_bencher.eval.metrics import bow_cosine, context_recall, lexical_f1

pytestmark = pytest.mark.unit


def test_metrics_sanity() -> None:
    a = "LangChain is a framework for LLM apps"
    b = "LangChain framework for language model applications"
    f1 = lexical_f1(a, b)
    cos = bow_cosine(a, b)
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= cos <= 1.0

    ctx = "This text mentions LangChain and language model apps."
    rec = context_recall(b, ctx)
    assert 0.0 <= rec <= 1.0


def test_metrics_handle_empty_inputs() -> None:
    assert lexical_f1("", "reference answer") == 0.0
    assert lexical_f1("prediction", "") == 0.0
    assert bow_cosine("", "reference answer") == 0.0
    assert bow_cosine("prediction", "") == 0.0
    assert context_recall("", "") == 0.0
