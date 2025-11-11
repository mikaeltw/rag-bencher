from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Type

if TYPE_CHECKING:
    from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


def _require() -> Type["VectorSearchVectorStore"]:
    try:
        from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

        return VectorSearchVectorStore
    except Exception as e:
        raise RuntimeError("Matching Engine requires langchain-google-vertexai (install rag-bench[gcp])") from e


class MatchingEngineBackend:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def make_retriever(
        self,
        *,
        docs: Optional[List[Document]],
        embeddings: Embeddings,
        k: int,
    ) -> VectorStoreRetriever:
        vector_store_cls = _require()
        proj = self.cfg.get("project_id")
        loc = self.cfg.get("location", "us-central1")
        idx = self.cfg.get("index_id")
        ep = self.cfg.get("endpoint_id")
        bucket = self.cfg.get("gcs_bucket_name")
        if not (proj and idx and ep and bucket):
            raise ValueError("project_id/index_id/endpoint_id/gcs_bucket_name required")
        vs = vector_store_cls.from_components(
            project_id=str(proj),
            region=str(loc),
            index_id=str(idx),
            endpoint_id=str(ep),
            gcs_bucket_name=str(bucket),
            embedding=embeddings,
        )
        return vs.as_retriever(search_kwargs={"k": k})
