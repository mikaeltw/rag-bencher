from typing import Any, Mapping, Optional, Protocol


class ChatAdapter(Protocol):
    def to_langchain(self) -> Any:
        """Return an initialized LangChain chat adapter."""
        ...


class EmbeddingsAdapter(Protocol):
    def to_langchain(self) -> Any:
        """Return an initialized LangChain embeddings adapter."""
        ...


def build_chat_adapter(cfg: Mapping[str, Any] | None) -> Optional[ChatAdapter]:
    if not cfg:
        return None
    name = (cfg.get("name") or "").lower()
    chat = cfg.get("chat", {})
    if name == "gcp":
        from .gcp.chat import VertexChatAdapter

        return VertexChatAdapter(chat)
    if name == "aws":
        from .aws.chat import BedrockChatAdapter

        return BedrockChatAdapter(cfg, chat)
    if name == "azure":
        from .azure.chat import AzureOpenAIChatAdapter

        return AzureOpenAIChatAdapter(chat)
    raise ValueError(f"Unknown provider: {name}")


def build_embeddings_adapter(cfg: Mapping[str, Any] | None) -> Optional[EmbeddingsAdapter]:
    if not cfg:
        return None
    name = (cfg.get("name") or "").lower()
    emb = cfg.get("embeddings", {})
    if name == "gcp":
        from .gcp.embeddings import VertexEmbeddingsAdapter

        return VertexEmbeddingsAdapter(emb)
    if name == "aws":
        from .aws.embeddings import BedrockEmbeddingsAdapter

        return BedrockEmbeddingsAdapter(cfg, emb)
    if name == "azure":
        from .azure.embeddings import AzureOpenAIEmbeddingsAdapter

        return AzureOpenAIEmbeddingsAdapter(emb)
    raise ValueError(f"Unknown provider: {name}")
