from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_bencher import bench_cli

pytestmark = [pytest.mark.unit, pytest.mark.offline]


class DummyChain:
    def __init__(self) -> None:
        self.calls: List[str] = []
        self.last_question: str = ""

    def invoke(self, question: str) -> str:
        self.last_question = question
        self.calls.append(question)
        return f"answer:{question}"


def _dummy_config() -> Any:
    class DummyCfg:
        def __init__(self) -> None:
            self.model = SimpleNamespace(name="demo-model")
            self.data = SimpleNamespace(paths=["doc.txt"])

        def model_dump(self) -> Dict[str, Any]:
            return {"model": {"name": self.model.name}}

    return DummyCfg()


def test_bench_cli_main_produces_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_entries = [
        {"question": "Q1", "reference_answer": "Ref1"},
        {"question": "Q2", "reference_answer": "Ref2"},
    ]
    qa_path.write_text("\n".join(json.dumps(e) for e in qa_entries), encoding="utf-8")

    cfg = _dummy_config()
    docs = ["doc"]
    docs_called: list[list[str]] = []
    monkeypatch.setattr(bench_cli, "load_config", lambda path: cfg)

    def fake_load_texts(paths: list[str]) -> list[str]:
        docs_called.append(paths)
        return docs

    monkeypatch.setattr(bench_cli, "load_texts_as_documents", fake_load_texts)

    chain = DummyChain()

    def debug() -> Dict[str, Any]:
        if chain.last_question == "Q1":
            return {
                "pipeline": "naive",
                "retrieved": [{"preview": f"ctx:{chain.last_question}", "source": "doc"}],
            }
        return {
            "pipeline": "naive",
            "candidates": [{"preview": f"cand:{chain.last_question}", "source": "doc"}],
        }

    selection = SimpleNamespace(
        pipeline_id="naive",
        chain=chain,
        debug=debug,
        config=cfg,
    )
    monkeypatch.setattr(bench_cli, "select_pipeline", lambda *_args, **_kwargs: selection)

    reports: List[Any] = []

    def fake_write_simple_report(**kwargs: Any) -> str:
        reports.append(kwargs)
        return "reports/report-123.html"

    monkeypatch.setattr(bench_cli, "write_simple_report", fake_write_simple_report)
    monkeypatch.setattr(sys, "argv", ["bench_cli", "--config", "cfg.yaml", "--qa", str(qa_path)])

    bench_cli.main()

    assert chain.calls == ["Q1", "Q2"]
    assert docs_called == [cfg.data.paths]
    assert reports, "report should be recorded"
    assert reports[0]["extras"] == {"pipeline": "naive"}


def test_bench_cli_uses_candidates_when_no_retrieved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question":"Q1","reference_answer":"Ref"}\n', encoding="utf-8")
    cfg = _dummy_config()
    chain = DummyChain()
    selection = SimpleNamespace(
        pipeline_id="naive",
        chain=chain,
        debug=lambda: {"pipeline": "naive", "candidates": [{"preview": "cand:Q1", "source": "doc"}]},
        config=cfg,
    )
    monkeypatch.setattr(bench_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(bench_cli, "load_texts_as_documents", lambda _: ["doc"])
    monkeypatch.setattr(bench_cli, "select_pipeline", lambda *_args, **_kwargs: selection)
    monkeypatch.setattr(bench_cli, "write_simple_report", lambda **_: "reports/report.html")
    monkeypatch.setattr(sys, "argv", ["bench_cli", "--config", "cfg.yaml", "--qa", str(qa_path)])

    bench_cli.main()

    assert chain.calls == ["Q1"]


def test_bench_cli_no_debug_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question":"Q1","reference_answer":"Ref"}\n', encoding="utf-8")
    cfg = _dummy_config()
    chain = DummyChain()
    selection = SimpleNamespace(
        pipeline_id="naive",
        chain=chain,
        debug=lambda: {"pipeline": "naive"},
        config=cfg,
    )
    reports: list[Any] = []
    monkeypatch.setattr(bench_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(bench_cli, "load_texts_as_documents", lambda _: ["doc"])
    monkeypatch.setattr(bench_cli, "select_pipeline", lambda *_args, **_kwargs: selection)

    def capture_report(**kwargs: Any) -> str:
        reports.append(kwargs)
        return "reports/report.html"

    monkeypatch.setattr(bench_cli, "write_simple_report", capture_report)
    monkeypatch.setattr(sys, "argv", ["bench_cli", "--config", "cfg.yaml", "--qa", str(qa_path)])

    bench_cli.main()

    assert chain.calls == ["Q1"]
    assert reports, "report should be generated even without context"
