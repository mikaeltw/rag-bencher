import os

import pytest

pytestmark = pytest.mark.cloud
RUN = os.getenv("RUN_GCP_SMOKE") == "true"


@pytest.mark.skipif(not RUN, reason="GCP smoke disabled")
def test_vertex_chat_smoke() -> None:
    from langchain_google_vertexai import ChatVertexAI

    llm = ChatVertexAI(
        model=os.getenv("VERTEX_CHAT_MODEL", "gemini-1.5-pro"),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
        project=os.getenv("GCP_PROJECT"),
        temperature=0,
    )
    out = llm.invoke("Say 'pong'")
    text = str(getattr(out, "content", out))
    assert "pong" in text.lower()
