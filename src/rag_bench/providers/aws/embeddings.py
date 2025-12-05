from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .auth import is_installed

if TYPE_CHECKING:
    from langchain_aws import BedrockEmbeddings


class BedrockEmbeddingsAdapter:
    def __init__(self, root_cfg: Mapping[str, Any], cfg: Mapping[str, Any]):
        self.root = root_cfg
        self.cfg = cfg

    def to_langchain(self) -> "BedrockEmbeddings":
        if not is_installed():
            raise RuntimeError("Install: rag-bencher[aws]")
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=self.cfg.get("model", "amazon.titan-embed-text-v2:0"),
            region_name=self.root.get("region", "us-east-1"),
        )
