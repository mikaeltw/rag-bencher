import os

from inspect import signature
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.cloud
RUN = os.getenv("RUN_AWS_SMOKE") == "true"


@pytest.mark.skipif(not RUN, reason="AWS smoke disabled")
def test_bedrock_chat_smoke() -> None:
    from langchain_aws import ChatBedrock

    params = signature(ChatBedrock).parameters
    kwargs: Dict[str, Any] = {"temperature": 0}
    model = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    if "model" in params:
        kwargs["model"] = model
    else:
        kwargs["model_id"] = model

    region = os.getenv("AWS_REGION", "us-east-1")
    if "region_name" in params:
        kwargs["region_name"] = region
    elif "client" in params:
        import boto3

        kwargs["client"] = boto3.client("bedrock-runtime", region_name=region)

    llm = ChatBedrock(**kwargs)
    out = llm.invoke("Say 'pong'")
    text = str(getattr(out, "content", out))
    assert "pong" in text.lower()
