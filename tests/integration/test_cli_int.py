import sys
from pathlib import Path
from typing import Any, cast

import pytest

from rag_bencher import cli
from rag_bencher.config import BenchConfig, DataCfg, ModelCfg, RetrieverCfg


class _DummyChain:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, *_args: Any, **_kwargs: Any) -> str:  # pragma: no cover - exercised via CLI call
        return self.answer


@pytest.mark.integration
@pytest.mark.gpu
def test_cli_entrypoint_emits_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    cfg = BenchConfig(
        model=ModelCfg(name="dummy-model"),
        retriever=RetrieverCfg(k=1),
        data=DataCfg(paths=[]),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model:\n  name: dummy-model\nretriever:\n  k: 1\ndata:\n  paths: []", encoding="utf-8")

    monkeypatch.setenv("RAG_BENCH_DEVICE", "cuda")
    monkeypatch.setattr(cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr(cli, "load_texts_as_documents", lambda _paths: [])
    monkeypatch.setattr(cli, "_pick_llm", lambda _cfg: None)
    monkeypatch.setattr(
        cast(Any, cli).naive_rag,
        "build_chain",
        lambda *_args, **_kwargs: (_DummyChain("stubbed answer"), lambda: {}),
    )
    monkeypatch.setattr(cli, "cache_get", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["rag-bencher-cli", "--config", str(cfg_path), "--question", "Hello?"])

    cli.main()

    captured = capsys.readouterr()
    assert "stubbed answer" in captured.out
