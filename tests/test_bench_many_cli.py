from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_bencher import bench_many_cli

pytestmark = [pytest.mark.unit, pytest.mark.offline]


class DummyChain:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: List[str] = []

    def invoke(self, question: str) -> str:
        self.calls.append(question)
        return f"{self.tag}:{question}"


def _selection(tag: str, cfg: Any, *, retrieved: bool) -> Any:
    chain = DummyChain(tag)

    def debug() -> Dict[str, Any]:
        payload: Dict[str, Any] = {"pipeline": tag}
        if retrieved:
            payload["retrieved"] = [{"source": "doc", "preview": f"{tag}-ctx"}]
        else:
            payload["candidates"] = [{"source": "doc", "preview": f"{tag}-cand", "score": 0.5}]
        return payload

    return SimpleNamespace(pipeline_id=f"pipe-{tag}", chain=chain, debug=debug, config=cfg)


def test_bench_many_cli_builds_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    qa_path = tmp_path / "qa.jsonl"
    qa_entries = [
        {"question": "Q1", "reference_answer": "Ref1"},
        {"question": "Q2", "reference_answer": "Ref2"},
    ]
    qa_path.write_text("\n".join(json.dumps(e) for e in qa_entries), encoding="utf-8")

    cfg = SimpleNamespace(
        model=SimpleNamespace(name="demo-model"),
        data=SimpleNamespace(paths=["doc.txt"]),
        retriever=SimpleNamespace(k=2),
        model_dump=lambda: {"model": {"name": "demo-model"}},
    )
    configs = [tmp_path / "cfg-a.yaml", tmp_path / "cfg-b.yaml"]
    for path in configs:
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(bench_many_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(bench_many_cli, "load_texts_as_documents", lambda paths: ["doc"])

    selections = {
        str(configs[0]): _selection("first", cfg, retrieved=True),
        str(configs[1]): _selection("second", cfg, retrieved=False),
    }
    monkeypatch.setattr(bench_many_cli, "select_pipeline", lambda path, docs: selections[path])

    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_many_cli", "--configs", str(tmp_path / "cfg-*.yaml"), "--qa", str(qa_path)],
    )

    bench_many_cli.main()

    reports_dir = Path("reports")
    outputs = list(reports_dir.glob("summary-*.html"))
    assert outputs, "expected summary file"
    html = outputs[0].read_text(encoding="utf-8")
    assert "cfg-a.yaml" in html and "cfg-b.yaml" in html
    assert "pipe-first" in html and "pipe-second" in html


def test_bench_many_cli_handles_candidate_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question":"Q","reference_answer":"R"}\n', encoding="utf-8")

    cfg = SimpleNamespace(
        model=SimpleNamespace(name="demo-model"),
        data=SimpleNamespace(paths=["doc.txt"]),
        retriever=SimpleNamespace(k=2),
        model_dump=lambda: {"model": {"name": "demo-model"}},
    )
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("{}", encoding="utf-8")

    chain = DummyChain("cand")

    def debug() -> Dict[str, Any]:
        return {"pipeline": "cand", "candidates": [{"preview": "cand-preview", "source": "doc"}]}

    selection = SimpleNamespace(pipeline_id="pipe-cand", chain=chain, debug=debug, config=cfg)

    monkeypatch.setattr(bench_many_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(bench_many_cli, "load_texts_as_documents", lambda _: ["doc"])
    monkeypatch.setattr(bench_many_cli, "select_pipeline", lambda *_args, **_kwargs: selection)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_many_cli", "--configs", str(config_path), "--qa", str(qa_path)],
    )

    bench_many_cli.main()

    assert chain.calls == ["Q"]


def test_bench_many_cli_handles_empty_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question":"Q","reference_answer":"R"}\n', encoding="utf-8")
    cfg = SimpleNamespace(
        model=SimpleNamespace(name="demo-model"),
        data=SimpleNamespace(paths=["doc.txt"]),
        retriever=SimpleNamespace(k=2),
        model_dump=lambda: {"model": {"name": "demo-model"}},
    )
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("{}", encoding="utf-8")
    chain = DummyChain("none")
    selection = SimpleNamespace(pipeline_id="pipe-none", chain=chain, debug=lambda: {"pipeline": "none"}, config=cfg)
    monkeypatch.setattr(bench_many_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(bench_many_cli, "load_texts_as_documents", lambda _: ["doc"])
    monkeypatch.setattr(bench_many_cli, "select_pipeline", lambda *_args, **_kwargs: selection)
    monkeypatch.setattr(sys, "argv", ["bench_many_cli", "--configs", str(config_path), "--qa", str(qa_path)])

    bench_many_cli.main()

    assert chain.calls == ["Q"]
