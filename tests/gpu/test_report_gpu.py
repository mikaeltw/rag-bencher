from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher.eval import report

pytestmark = pytest.mark.gpu


def test_render_extras_empty_returns_blank() -> None:
    assert report._render_extras({}) == ""


def test_render_extras_gpu_branches() -> None:
    extras = {
        "pipeline": "multi",
        "retrieved": [{"source": "doc", "preview": "<alpha>ctx</alpha>"}],
        "candidates": [{"source": "doc", "preview": "<beta>cand</beta>", "score": 0.42}],
    }
    html = report._render_extras(extras)
    assert "Retrieved snippets" in html
    assert "Rerank candidates" in html
    assert "&lt;alpha&gt;ctx" in html


def test_write_simple_report_without_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = report.write_simple_report("Q?", "A", cfg={"model": "demo"}, extras=None)
    saved = Path(path)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "rag-bencher report" in content
