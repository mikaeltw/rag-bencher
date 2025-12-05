import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

_VectorStoreFactory = type[VectorStore]


def build_local_vectorstore(documents: Iterable[Document], embeddings: Embeddings) -> VectorStore:
    """Construct a local vector store with FAISS when possible and a safe fallback otherwise."""
    factory = _resolve_factory()
    doc_list = list(documents)
    return factory.from_documents(doc_list, embeddings)


@lru_cache(maxsize=1)
def _resolve_factory() -> _VectorStoreFactory:
    mode = (os.getenv("RAG_BENCH_VECTORSTORE") or "auto").strip().lower()
    disable_faiss = _is_truthy(os.getenv("RAG_BENCH_DISABLE_FAISS"))

    if mode in {"memory", "inmemory", "in-memory"} or disable_faiss:
        return _inmemory_factory()

    if mode == "faiss":
        if not _faiss_safe_to_import():
            raise RuntimeError(
                "RAG_BENCH_VECTORSTORE=faiss but FAISS is unavailable or unsafe to import in this environment."
            )
        return _faiss_factory()

    # Default to the safe in-memory implementation unless FAISS is explicitly requested.
    if mode not in {"", "auto"}:
        raise ValueError(f"Unknown RAG_BENCH_VECTORSTORE={mode!r}. Expected faiss or memory.")
    return _inmemory_factory()


def _faiss_factory() -> _VectorStoreFactory:
    from langchain_community.vectorstores.faiss import FAISS

    return FAISS


def _inmemory_factory() -> _VectorStoreFactory:
    from langchain_community.vectorstores.inmemory import InMemoryVectorStore

    return InMemoryVectorStore


@lru_cache(maxsize=1)
def _faiss_safe_to_import() -> bool:
    """Check FAISS availability without risking a segfault in the current process."""
    spec = importlib.util.find_spec("faiss")
    if spec is None:
        return False

    probe = "from langchain_community.vectorstores.faiss import FAISS"  # noqa: F401 - import test only
    try:
        subprocess.run(
            [sys.executable or "python", "-c", probe],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False

    return True


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
