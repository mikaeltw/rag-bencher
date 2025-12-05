from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher.eval import dataset_loader
from rag_bencher.eval import datasets as dataset_mod

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_load_texts_as_documents(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_a.write_text("alpha", encoding="utf-8")
    docs = dataset_loader.load_texts_as_documents([str(file_a)])
    assert len(docs) == 1
    assert docs[0].page_content == "alpha"
    assert docs[0].metadata == {"source": str(file_a)}


def test_list_and_load_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    (root / "set1").mkdir(parents=True)
    (root / "set1" / "doc.txt").write_text("hello", encoding="utf-8")
    (root / "set2").mkdir()
    (root / "set2" / "doc.md").write_text("world", encoding="utf-8")
    monkeypatch.setattr(dataset_mod, "DATASETS_ROOT", root, raising=False)

    names = dataset_mod.list_datasets()
    assert names == ["set1", "set2"]

    docs = dataset_mod.load_dataset("set1")
    assert len(docs) == 1
    assert docs[0].metadata["source"].endswith("doc.txt")
    assert "hello" in docs[0].page_content


def test_load_dataset_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(dataset_mod, "DATASETS_ROOT", root, raising=False)
    with pytest.raises(FileNotFoundError):
        dataset_mod.load_dataset("missing")


def test_load_dataset_without_text_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    (root / "empty").mkdir(parents=True)
    monkeypatch.setattr(dataset_mod, "DATASETS_ROOT", root, raising=False)
    with pytest.raises(FileNotFoundError):
        dataset_mod.load_dataset("empty")
