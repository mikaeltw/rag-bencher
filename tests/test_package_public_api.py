from __future__ import annotations

import importlib
import sys
import types

import pytest

import rag_bencher

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_package_exports_version_and_loader() -> None:
    assert isinstance(rag_bencher.__version__, str)
    assert "load_config" in rag_bencher.__all__
    assert hasattr(rag_bencher, "load_config")


def test_package_handles_missing_config_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("rag_bencher.config")
    monkeypatch.setitem(sys.modules, "rag_bencher.config", fake_mod)
    module = importlib.reload(rag_bencher)
    assert "load_config" not in module.__all__
    assert not hasattr(module, "load_config")

    # Restore the real module for the rest of the suite.
    monkeypatch.undo()
    importlib.reload(rag_bencher)
