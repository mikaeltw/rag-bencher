from __future__ import annotations

import importlib
import importlib.metadata
from typing import Iterator

import pytest

import rag_bencher

pytestmark = pytest.mark.gpu


def _reload_rag_bencher() -> None:
    importlib.reload(rag_bencher)


@pytest.fixture(autouse=True)
def _reset_module() -> Iterator[None]:
    yield
    importlib.reload(rag_bencher)


def test_import_sets_effective_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_bencher.utils.hardware.apply_process_wide_policy", lambda: "cuda")
    _reload_rag_bencher()
    assert rag_bencher._EFFECTIVE_DEVICE == "cuda"


def test_import_handles_device_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise RuntimeError("no device")

    monkeypatch.setattr("rag_bencher.utils.hardware.apply_process_wide_policy", boom)
    _reload_rag_bencher()
    assert rag_bencher._EFFECTIVE_DEVICE == "auto"


def test_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    _reload_rag_bencher()
    assert rag_bencher.__version__ == "0.0.0"


def test_reload_appends_load_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    original_cfg = sys.modules.get("rag_bencher.config")
    monkeypatch.delitem(sys.modules, "rag_bencher.config", raising=False)
    module = importlib.reload(rag_bencher)
    assert "load_config" in module.__all__
    if original_cfg is not None:
        monkeypatch.setitem(sys.modules, "rag_bencher.config", original_cfg)
    importlib.reload(rag_bencher)
