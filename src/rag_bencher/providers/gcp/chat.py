from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .auth import is_installed

if TYPE_CHECKING:
    from langchain_google_vertexai import ChatVertexAI


class VertexChatAdapter:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def to_langchain(self) -> "ChatVertexAI":
        if not is_installed():
            raise RuntimeError("Install: rag-bencher[gcp]")
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=self.cfg.get("model", "gemini-1.5-pro"),
            location=self.cfg.get("location", "us-central1"),
            project=self.cfg.get("project_id"),
            temperature=0,
        )
