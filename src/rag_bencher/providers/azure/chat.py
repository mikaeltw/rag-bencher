from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .auth import is_installed

if TYPE_CHECKING:
    from langchain_openai import AzureChatOpenAI


class AzureOpenAIChatAdapter:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def to_langchain(self) -> "AzureChatOpenAI":
        if not is_installed():
            raise RuntimeError("Install: rag-bencher[azure]")
        from langchain_openai import AzureChatOpenAI

        dep = str(self.cfg.get("deployment", "gpt-4o-mini"))
        endpoint = self.cfg.get("endpoint")
        ver = str(self.cfg.get("api_version", "2024-06-01"))
        if not endpoint:
            raise ValueError("Azure OpenAI requires endpoint")
        return AzureChatOpenAI(azure_deployment=dep, azure_endpoint=str(endpoint), api_version=ver, temperature=0)
