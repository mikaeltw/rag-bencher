from typing import Any, Dict, List, Optional, Protocol

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorBackend(Protocol):
    def make_retriever(self, *, docs: Optional[List[Document]], embeddings: Any, k: int) -> BaseRetriever:
        """Return a retriever backed by the configured vector store."""
        ...


def build_vector_backend(cfg: Dict[str, Any] | None) -> Optional[VectorBackend]:
    if not cfg:
        return None
    name = (cfg.get("name") or "").lower()
    if name == "azure_ai_search":
        from .azure_ai_search import AzureAISearchBackend

        return AzureAISearchBackend(cfg)
    if name == "opensearch":
        from .opensearch import OpenSearchBackend

        return OpenSearchBackend(cfg)
    if name == "matching_engine":
        from .matching_engine import MatchingEngineBackend

        return MatchingEngineBackend(cfg)
    raise ValueError(f"Unknown vector backend: {name}")
