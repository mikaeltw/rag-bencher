from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Mapping

from .auth import is_installed

if TYPE_CHECKING:
    from langchain_aws import ChatBedrock


class BedrockChatAdapter:
    def __init__(self, root_cfg: Mapping[str, Any], cfg: Mapping[str, Any]):
        self.root = root_cfg
        self.cfg = cfg

    def to_langchain(self) -> "ChatBedrock":
        if not is_installed():
            raise RuntimeError("Install: rag-bencher[aws]")
        from langchain_aws import ChatBedrock

        kwargs: dict[str, Any] = {"temperature": self.cfg.get("temperature", 0)}
        params = signature(ChatBedrock).parameters
        model = self.cfg.get("model", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        if "model" in params:
            kwargs["model"] = model
        elif "model_id" in params:
            kwargs["model_id"] = model
        else:  # pragma: no cover - defensive against unexpected API changes
            raise RuntimeError("ChatBedrock requires either 'model' or 'model_id'")

        region = self.root.get("region", "us-east-1")
        if "region_name" in params:
            kwargs["region_name"] = region
        elif "client" in params:
            import boto3

            kwargs["client"] = boto3.client("bedrock-runtime", region_name=region)

        return ChatBedrock(**kwargs)
