from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag_bencher.eval.dataset_loader import load_texts_as_documents

DATASETS_ROOT = Path("examples/datasets")


def list_datasets() -> list[str]:
    if not DATASETS_ROOT.exists():
        return []
    out = []
    for p in DATASETS_ROOT.rglob("*"):
        if p.is_dir() and any(p.glob("*.txt")) or any(p.glob("*.md")):
            rel = p.relative_to(DATASETS_ROOT).as_posix()
            out.append(rel)
    return sorted(set(out))


def load_dataset(name: str) -> list[Document]:
    """Load a dataset by name like 'docs/wiki' -> examples/datasets/docs/wiki/*.txt|*.md."""
    path = DATASETS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset {name!r} not found under {DATASETS_ROOT}")
    files = [str(p) for p in path.glob("*.txt")] + [str(p) for p in path.glob("*.md")]
    if not files:
        raise FileNotFoundError(f"No .txt or .md files in dataset {name}")
    return load_texts_as_documents(files)
