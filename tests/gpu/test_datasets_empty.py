from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher.eval import datasets

pytestmark = pytest.mark.gpu


def test_list_datasets_handles_missing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_root = tmp_path / "no-datasets"
    monkeypatch.setattr(datasets, "DATASETS_ROOT", missing_root, raising=False)
    assert datasets.list_datasets() == []
