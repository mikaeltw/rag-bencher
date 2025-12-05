import sys
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableSerializable

from rag_bencher import bench_many_cli
from rag_bencher.config import BenchConfig, DataCfg, ModelCfg, RetrieverCfg
from rag_bencher.pipelines.selector import PipelineSelection


class _DummyChain:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, *_args: Any, **_kwargs: Any) -> str:  # pragma: no cover - exercised via CLI call
        return self.answer


@pytest.mark.integration
@pytest.mark.gpu
def test_bench_many_cli_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = BenchConfig(
        model=ModelCfg(name="dummy-model"),
        retriever=RetrieverCfg(k=1),
        data=DataCfg(paths=[]),
    )
    cfg1 = tmp_path / "cfg1.yaml"
    cfg2 = tmp_path / "cfg2.yaml"
    for cfg_file in (cfg1, cfg2):
        cfg_file.write_text("model:\n  name: dummy-model\nretriever:\n  k: 1\ndata:\n  paths: []", encoding="utf-8")

    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question": "Q?", "reference_answer": "A"}\n', encoding="utf-8")

    def _debug() -> dict[str, list[dict[str, str]]]:
        return {"retrieved": [{"preview": "context"}]}

    def _select(path: str, _docs: list[object], _cfg: BenchConfig | None = None) -> PipelineSelection:
        return PipelineSelection(
            pipeline_id=Path(path).stem,
            config=cfg,
            chain=cast(RunnableSerializable[str, str], _DummyChain("A")),
            debug=_debug,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_BENCH_DEVICE", "cuda")
    monkeypatch.setattr(bench_many_cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr(bench_many_cli, "load_texts_as_documents", lambda _paths: [])
    monkeypatch.setattr(bench_many_cli, "select_pipeline", _select)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-bencher-cli-bench-many",
            "--configs",
            str(tmp_path / "cfg*.yaml"),
            "--qa",
            str(qa_path),
        ],
    )

    bench_many_cli.main()

    captured = capsys.readouterr()
    assert "cfg1.yaml" in captured.out
    summary_files = list((tmp_path / "reports").glob("summary-*.html"))
    assert summary_files, "summary report was not created"
