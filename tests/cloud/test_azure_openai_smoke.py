import os

import pytest
from langchain_core.messages import BaseMessage

pytestmark = pytest.mark.cloud
RUN = os.getenv("RUN_AZURE_SMOKE") == "true"


@pytest.mark.skipif(not RUN, reason="Azure smoke disabled")
def test_azure_openai_chat_smoke() -> None:
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("Missing Azure env")
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        temperature=0,
    )
    out = llm.invoke("Say 'pong'")
    assert isinstance(out, BaseMessage)
    raw = out.content
    if isinstance(raw, str):
        content = raw
    else:
        content = " ".join(str(part) for part in raw)
    assert "pong" in content.lower()
