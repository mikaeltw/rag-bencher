from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Mapping

from .auth import is_installed

if TYPE_CHECKING:
    from langchain_google_vertexai import VertexAIEmbeddings


class VertexEmbeddingsAdapter:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def to_langchain(self) -> "VertexAIEmbeddings":
        if not is_installed():
            raise RuntimeError("Install: rag-bencher[gcp]")
        from langchain_google_vertexai import VertexAIEmbeddings

        params = signature(VertexAIEmbeddings).parameters
        kwargs: dict[str, Any] = {
            "location": self.cfg.get("location", "us-central1"),
            "project": self.cfg.get("project_id"),
        }
        model = self.cfg.get("model", "text-embedding-004")
        if "model" in params:
            kwargs["model"] = model
        elif "model_name" in params:
            kwargs["model_name"] = model
        else:  # pragma: no cover - defensive against unexpected API changes
            raise RuntimeError("VertexAIEmbeddings requires 'model' or 'model_name'")

        return VertexAIEmbeddings(**kwargs)
