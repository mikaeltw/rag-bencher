from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher.eval import report


@pytest.mark.offline
def test_render_extras_renders_all_sections() -> None:
    extras = {
        "pipeline": "multi_query",
        "method": "cosine",
        "run_id": "abc123",
        "queries": ["How?", "Why?"],
        "retrieved": [{"source": "doc1", "preview": "<highlight>snippet</highlight>"}],
        "candidates": [{"source": "doc2", "preview": "<tag>c</tag>", "score": 0.98765}],
        "usage": {"prompt_tokens": 12},
    }
    html = report._render_extras(extras)
    assert "Debug" in html
    assert "Generated sub-queries" in html
    assert "&lt;highlight&gt;snippet" in html  # ensures escaping
    assert "<td>0.9877</td>" in html
    assert "<h3>Usage</h3>" in html


@pytest.mark.offline
def test_render_extras_without_retrieved() -> None:
    html = report._render_extras({"pipeline": "naive"})
    assert "Retrieved snippets" not in html


@pytest.mark.offline
def test_write_simple_report_creates_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    extras = {"retrieved": [{"source": "doc", "preview": "sample"}]}
    path = report.write_simple_report(
        question="Who?",
        answer="Answer",
        cfg={"model": {"name": "demo"}},
        extras=extras,
    )
    output = Path(path)
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "rag-bencher report" in html
    assert "Who?" in html
    assert "Answer" in html
    assert "retrieved" in html.lower()
