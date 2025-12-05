from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Type, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

if TYPE_CHECKING:
    from langchain_community.vectorstores.azuresearch import AzureSearch


def _require() -> Type["AzureSearch"]:
    try:
        from langchain_community.vectorstores.azuresearch import AzureSearch

        return AzureSearch
    except Exception as e:
        raise RuntimeError("Azure AI Search requires azure-search-documents (install rag-bencher[azure])") from e


class AzureAISearchBackend:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def make_retriever(
        self,
        *,
        docs: Optional[List[Document]],
        embeddings: Embeddings,
        k: int,
    ) -> VectorStoreRetriever:
        AzureSearch = _require()
        ep = self.cfg.get("endpoint")
        idx = self.cfg.get("index")
        key = self.cfg.get("api_key")
        if not ep or not idx:
            raise ValueError("endpoint/index required")
        vs = AzureSearch(azure_search_endpoint=ep, azure_search_key=key, index_name=idx, embedding_function=embeddings)
        return cast(VectorStoreRetriever, vs.as_retriever(search_kwargs={"k": k}))
