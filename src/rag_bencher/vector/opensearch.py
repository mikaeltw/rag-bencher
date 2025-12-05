from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

if TYPE_CHECKING:
    from langchain_community.vectorstores import OpenSearchVectorSearch


def _require() -> Type["OpenSearchVectorSearch"]:
    try:
        from langchain_community.vectorstores import OpenSearchVectorSearch

        return OpenSearchVectorSearch
    except Exception as e:
        raise RuntimeError("OpenSearch requires opensearch-py (install rag-bencher[aws])") from e


class OpenSearchBackend:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def make_retriever(
        self,
        *,
        docs: Optional[List[Document]],
        embeddings: Embeddings,
        k: int,
    ) -> VectorStoreRetriever:
        V = _require()
        hosts = self.cfg.get("hosts")
        opensearch_url = self.cfg.get("opensearch_url")
        idx = self.cfg.get("index")
        if not idx:
            raise ValueError("index required")
        if not opensearch_url and not hosts:
            raise ValueError("opensearch_url or hosts required")
        if not opensearch_url and hosts:
            if isinstance(hosts, list) and hosts:
                opensearch_url = hosts[0]
            else:
                opensearch_url = hosts
        if not isinstance(opensearch_url, str):
            raise ValueError("opensearch_url must resolve to a string")
        vs = V(
            opensearch_url=opensearch_url,
            index_name=idx,
            embedding_function=embeddings,
            hosts=hosts,
            use_ssl=self.cfg.get("use_ssl", True),
            verify_certs=self.cfg.get("verify_certs", True),
            http_auth=self.cfg.get("http_auth"),
        )
        return vs.as_retriever(search_kwargs={"k": k})
