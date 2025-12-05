import os
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_openai import ChatOpenAI


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def resolve_chat_llm(
    model: str,
    *,
    temperature: float = 0.0,
    override: Optional[RunnableSerializable[Any, Any]] = None,
) -> RunnableSerializable[Any, Any]:
    """Return the provided LLM, an OpenAI chat model, or a simple local fallback."""
    if override is not None:
        return override
    if has_openai_key():
        return ChatOpenAI(model=model, temperature=temperature)

    def _offline(prompt: Any) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return f"[offline answer]\n{text[:400]}"

    offline_chain = RunnableLambda(_offline)
    return cast(RunnableSerializable[Any, Any], offline_chain)
