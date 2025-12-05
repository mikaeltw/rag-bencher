import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Tuple, cast

import pytest
from langchain_core.runnables import RunnableSerializable

from rag_bencher.config import BenchConfig, DataCfg, ModelCfg, RetrieverCfg
from rag_bencher.pipelines.selector import PipelineSelection


class _DummyChain:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, *_args: Any, **_kwargs: Any) -> str:  # pragma: no cover - exercised via example call
        return self.answer


def _setup_example(monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, PipelineSelection]:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))

    import examples.compare_two_pipelines as ex

    cfg = BenchConfig(model=ModelCfg(name="stub"), retriever=RetrieverCfg(k=1), data=DataCfg(paths=[]))
    selection = PipelineSelection(
        pipeline_id="stub-pipeline",
        config=cfg,
        chain=cast(RunnableSerializable[str, str], _DummyChain("reference")),
        debug=lambda: {"retrieved": [{"preview": "context"}]},
    )

    monkeypatch.setattr(ex, "DOCS", ["doc"])
    monkeypatch.setattr(
        ex,
        "QA_EXAMPLES",
        [
            {"question": "Q1", "reference_answer": "reference"},
            {"question": "Q2", "reference_answer": "reference"},
        ],
    )
    monkeypatch.setattr(ex, "CONFIGS", ("cfg-a.yaml", "cfg-b.yaml"))
    monkeypatch.setattr(ex, "load_config", lambda _path: cfg)
    monkeypatch.setattr(ex, "select_pipeline", lambda *_args, **_kwargs: selection)
    return ex, selection


@pytest.mark.examples
@pytest.mark.offline
def test_evaluate_runs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    ex, selection = _setup_example(monkeypatch)
    monkeypatch.setenv("RAG_BENCH_DEVICE", "cpu")

    pid, metrics = ex.evaluate("cfg-a.yaml")

    assert pid == selection.pipeline_id
    assert set(metrics.keys()) == {"lexical_f1", "bow_cosine", "context_recall"}
    assert all(v >= 0.0 for v in metrics.values())


@pytest.mark.examples
@pytest.mark.gpu
def test_main_runs_on_gpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ex, selection = _setup_example(monkeypatch)
    monkeypatch.setenv("RAG_BENCH_DEVICE", "cuda")

    ex.main()

    out = capsys.readouterr().out
    assert "cfg-a.yaml" in out and "cfg-b.yaml" in out
    assert selection.pipeline_id in out


def _run_main_with_stubbed_modules(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    device: str,
    debug_payload: dict[str, Any],
) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))

    rb_pkg = ModuleType("rag_bencher")
    rb_pkg.__path__ = []
    rb_eval = ModuleType("rag_bencher.eval")
    rb_eval.__path__ = []
    rb_pipelines = ModuleType("rag_bencher.pipelines")
    rb_pipelines.__path__ = []

    cfg_mod = ModuleType("rag_bencher.config")
    cfg_mod.load_config = lambda _path: "cfg"  # type: ignore[attr-defined]

    dl_mod = ModuleType("rag_bencher.eval.dataset_loader")
    dl_mod.load_texts_as_documents = lambda _paths: ["doc"]  # type: ignore[attr-defined]

    metrics_mod = ModuleType("rag_bencher.eval.metrics")
    metrics_mod.lexical_f1 = lambda *_args, **_kwargs: 1.0  # type: ignore[attr-defined]
    metrics_mod.bow_cosine = lambda *_args, **_kwargs: 1.0  # type: ignore[attr-defined]
    metrics_mod.context_recall = lambda *_args, **_kwargs: 1.0  # type: ignore[attr-defined]

    selector_mod = ModuleType("rag_bencher.pipelines.selector")

    class _Sel:
        def __init__(self) -> None:
            self.pipeline_id = "stub"
            self.chain = cast(RunnableSerializable[str, str], _DummyChain("ans"))
            self.debug = lambda: debug_payload

    selector_mod.select_pipeline = lambda *_args, **_kwargs: _Sel()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rag_bencher", rb_pkg)
    monkeypatch.setitem(sys.modules, "rag_bencher.config", cfg_mod)
    monkeypatch.setitem(sys.modules, "rag_bencher.eval", rb_eval)
    monkeypatch.setitem(sys.modules, "rag_bencher.eval.dataset_loader", dl_mod)
    monkeypatch.setitem(sys.modules, "rag_bencher.eval.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "rag_bencher.pipelines", rb_pipelines)
    monkeypatch.setitem(sys.modules, "rag_bencher.pipelines.selector", selector_mod)

    monkeypatch.setenv("RAG_BENCH_DEVICE", device)
    sys.modules.pop("examples.compare_two_pipelines", None)
    runpy.run_module("examples.compare_two_pipelines", run_name="__main__")
    return capsys.readouterr().out


@pytest.mark.examples
@pytest.mark.offline
def test_evaluate_handles_candidate_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    ex, _selection = _setup_example(monkeypatch)
    cfg = BenchConfig(model=ModelCfg(name="stub"), retriever=RetrieverCfg(k=1), data=DataCfg(paths=[]))
    candidate_selection = PipelineSelection(
        pipeline_id="stub-pipeline",
        config=cfg,
        chain=cast(RunnableSerializable[str, str], _DummyChain("reference")),
        debug=lambda: {"candidates": [{"preview": "cand"}]},
    )
    monkeypatch.setattr(ex, "select_pipeline", lambda *_args, **_kwargs: candidate_selection)

    _, metrics = ex.evaluate("cfg-candidate.yaml")

    assert all(v >= 0.0 for v in metrics.values())


@pytest.mark.examples
@pytest.mark.offline
def test_evaluate_handles_missing_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    ex, _selection = _setup_example(monkeypatch)
    cfg = BenchConfig(model=ModelCfg(name="stub"), retriever=RetrieverCfg(k=1), data=DataCfg(paths=[]))
    missing_debug_selection = PipelineSelection(
        pipeline_id="stub-pipeline",
        config=cfg,
        chain=cast(RunnableSerializable[str, str], _DummyChain("reference")),
        debug=lambda: {},
    )
    monkeypatch.setattr(ex, "select_pipeline", lambda *_args, **_kwargs: missing_debug_selection)

    _, metrics = ex.evaluate("cfg-missing-debug.yaml")

    assert metrics["context_recall"] == 0.0
    assert all(v >= 0.0 for v in metrics.values())


@pytest.mark.examples
@pytest.mark.gpu
def test_script_runs_under_main_guard(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out = _run_main_with_stubbed_modules(
        monkeypatch,
        capsys,
        device="cuda",
        debug_payload={"retrieved": [{"preview": "ctx"}]},
    )
    assert "stub" in out


@pytest.mark.examples
@pytest.mark.offline
def test_script_runs_under_main_guard_cpu_candidates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _run_main_with_stubbed_modules(
        monkeypatch,
        capsys,
        device="cpu",
        debug_payload={"candidates": [{"preview": "cand"}]},
    )
    assert "stub" in out
    assert "context_recall" in out


@pytest.mark.examples
@pytest.mark.gpu
def test_script_runs_under_main_guard_gpu_candidates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _run_main_with_stubbed_modules(
        monkeypatch,
        capsys,
        device="cuda",
        debug_payload={"candidates": [{"preview": "cand"}]},
    )
    assert "stub" in out
    assert "context_recall" in out
