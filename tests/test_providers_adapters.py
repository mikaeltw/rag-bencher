from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from rag_bencher.providers import base
from rag_bencher.providers.aws.chat import BedrockChatAdapter
from rag_bencher.providers.aws.embeddings import BedrockEmbeddingsAdapter
from rag_bencher.providers.azure.chat import AzureOpenAIChatAdapter
from rag_bencher.providers.azure.embeddings import AzureOpenAIEmbeddingsAdapter
from rag_bencher.providers.gcp.chat import VertexChatAdapter
from rag_bencher.providers.gcp.embeddings import VertexEmbeddingsAdapter

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def _install_stub(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    def _installed(*_: Any, **__: Any) -> bool:
        return True

    monkeypatch.setattr(target, _installed)


def _fail_install(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    def _not_installed(*_: Any, **__: Any) -> bool:
        return False

    monkeypatch.setattr(target, _not_installed)


def test_bedrock_chat_adapter_uses_model_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyChat:
        def __init__(self, *, temperature: float, model: str, region_name: str) -> None:
            calls["kwargs"] = {"temperature": temperature, "model": model, "region_name": region_name}

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrock=DummyChat))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")

    adapter = BedrockChatAdapter({"region": "us-west-2"}, {"model": "anthropic.claude", "temperature": 0.25})
    llm = adapter.to_langchain()

    assert isinstance(llm, DummyChat)
    assert calls["kwargs"] == {"temperature": 0.25, "model": "anthropic.claude", "region_name": "us-west-2"}


def test_bedrock_chat_adapter_falls_back_to_model_id_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    boto: Dict[str, Any] = {}

    class DummyChat:
        def __init__(self, *, temperature: float, model_id: str, client: Any) -> None:
            calls["kwargs"] = {"temperature": temperature, "model_id": model_id, "client": client}

    def fake_client(service: str, region_name: str) -> str:
        boto["args"] = (service, region_name)
        return "bedrock-runtime-client"

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrock=DummyChat))
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")

    adapter = BedrockChatAdapter({"region": "eu-central-1"}, {"model": "anthropic.claude"})
    adapter.to_langchain()

    assert boto["args"] == ("bedrock-runtime", "eu-central-1")
    assert calls["kwargs"]["model_id"] == "anthropic.claude"
    assert calls["kwargs"]["client"] == "bedrock-runtime-client"


def test_bedrock_chat_adapter_skips_region_and_client_when_params_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Dict[str, Any] = {}

    class DummyChat:
        def __init__(self, *, temperature: float, model_id: str) -> None:
            calls["kwargs"] = {"temperature": temperature, "model_id": model_id}

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrock=DummyChat))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")

    adapter = BedrockChatAdapter({"region": "us-west-2"}, {"model": "anthropic.claude"})
    chat = adapter.to_langchain()

    assert isinstance(chat, DummyChat)
    assert calls["kwargs"] == {"temperature": 0, "model_id": "anthropic.claude"}


def test_bedrock_chat_adapter_builds_boto_client_when_region_param_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    boto: Dict[str, Any] = {}

    class DummyChat:
        def __init__(self, *, temperature: float, model_id: str, client: Any) -> None:
            self.kwargs = {"temperature": temperature, "model_id": model_id, "client": client}

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrock=DummyChat))
    monkeypatch.setattr(
        "rag_bencher.providers.aws.chat.signature",
        lambda _callable: SimpleNamespace(
            parameters={"temperature": object(), "model_id": object(), "client": object()}
        ),
    )

    def fake_client(service: str, region_name: str) -> str:
        boto["call"] = (service, region_name)
        return "runtime-client"

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")

    adapter = BedrockChatAdapter({"region": "sa-east-1"}, {"model": "anthropic.claude"})
    chat = adapter.to_langchain()

    assert isinstance(chat, DummyChat)
    assert boto["call"] == ("bedrock-runtime", "sa-east-1")
    assert chat.kwargs["client"] == "runtime-client"


def test_bedrock_chat_adapter_errors_when_model_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyChat:
        def __init__(self, *, temperature: float) -> None:  # pragma: no cover - invoked here
            pass

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrock=DummyChat))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")

    adapter = BedrockChatAdapter({}, {})
    with pytest.raises(RuntimeError, match="requires either 'model' or 'model_id'"):
        adapter.to_langchain()


def test_bedrock_chat_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.aws.chat.is_installed")
    adapter = BedrockChatAdapter({}, {})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[aws]"):
        adapter.to_langchain()


def test_bedrock_embeddings_adapter_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyEmbeddings:
        def __init__(self, *, model_id: str, region_name: str) -> None:
            calls["kwargs"] = {"model_id": model_id, "region_name": region_name}

    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(BedrockEmbeddings=DummyEmbeddings))
    _install_stub(monkeypatch, "rag_bencher.providers.aws.embeddings.is_installed")

    adapter = BedrockEmbeddingsAdapter({"region": "ap-south-1"}, {"model": "amazon.titan"})
    emb = adapter.to_langchain()

    assert isinstance(emb, DummyEmbeddings)
    assert calls["kwargs"] == {"model_id": "amazon.titan", "region_name": "ap-south-1"}


def test_bedrock_embeddings_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.aws.embeddings.is_installed")
    adapter = BedrockEmbeddingsAdapter({}, {})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[aws]"):
        adapter.to_langchain()


def test_azure_chat_adapter_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch, "rag_bencher.providers.azure.chat.is_installed")
    adapter = AzureOpenAIChatAdapter({"deployment": "gpt4"})
    with pytest.raises(ValueError, match="requires endpoint"):
        adapter.to_langchain()


def test_azure_chat_adapter_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyAzureChat:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(AzureChatOpenAI=DummyAzureChat))
    _install_stub(monkeypatch, "rag_bencher.providers.azure.chat.is_installed")

    adapter = AzureOpenAIChatAdapter(
        {"deployment": "gpt-4o-mini", "endpoint": "https://example", "api_version": "2024-05-01"}
    )
    chat = adapter.to_langchain()

    assert isinstance(chat, DummyAzureChat)
    assert calls["kwargs"] == {
        "azure_deployment": "gpt-4o-mini",
        "azure_endpoint": "https://example",
        "api_version": "2024-05-01",
        "temperature": 0,
    }


def test_azure_embeddings_adapter_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyAzureEmbeddings:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(AzureOpenAIEmbeddings=DummyAzureEmbeddings))
    _install_stub(monkeypatch, "rag_bencher.providers.azure.embeddings.is_installed")

    adapter = AzureOpenAIEmbeddingsAdapter(
        {"deployment": "text-embedding-3-large", "endpoint": "https://emb", "api_version": "2024-05-01"}
    )
    emb = adapter.to_langchain()

    assert isinstance(emb, DummyAzureEmbeddings)
    assert calls["kwargs"] == {
        "azure_deployment": "text-embedding-3-large",
        "azure_endpoint": "https://emb",
        "api_version": "2024-05-01",
    }


def test_azure_embeddings_adapter_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyAzureEmbeddings:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(AzureOpenAIEmbeddings=DummyAzureEmbeddings))
    _install_stub(monkeypatch, "rag_bencher.providers.azure.embeddings.is_installed")

    adapter = AzureOpenAIEmbeddingsAdapter({"endpoint": "https://default"})
    adapter.to_langchain()

    assert calls["kwargs"] == {
        "azure_deployment": "text-embedding-3-large",
        "azure_endpoint": "https://default",
        "api_version": "2024-06-01",
    }


def test_azure_embeddings_adapter_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAzureEmbeddings:
        pass

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(AzureOpenAIEmbeddings=DummyAzureEmbeddings))
    _install_stub(monkeypatch, "rag_bencher.providers.azure.embeddings.is_installed")

    adapter = AzureOpenAIEmbeddingsAdapter({})
    with pytest.raises(ValueError, match="requires endpoint"):
        adapter.to_langchain()


def test_azure_chat_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.azure.chat.is_installed")
    adapter = AzureOpenAIChatAdapter({"endpoint": "https://example"})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[azure]"):
        adapter.to_langchain()


def test_azure_embeddings_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.azure.embeddings.is_installed")
    adapter = AzureOpenAIEmbeddingsAdapter({"endpoint": "https://example"})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[azure]"):
        adapter.to_langchain()


def test_vertex_chat_adapter_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyVertexChat:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", SimpleNamespace(ChatVertexAI=DummyVertexChat))
    _install_stub(monkeypatch, "rag_bencher.providers.gcp.chat.is_installed")

    adapter = VertexChatAdapter({"model": "gemini", "location": "europe-west1", "project_id": "proj"})
    chat = adapter.to_langchain()

    assert isinstance(chat, DummyVertexChat)
    assert calls["kwargs"] == {"model": "gemini", "location": "europe-west1", "project": "proj", "temperature": 0}


def test_vertex_embeddings_adapter_supports_model_name_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyVertexEmbeddings:
        def __init__(self, *, location: str, project: str, model_name: str) -> None:
            calls["kwargs"] = {"location": location, "project": project, "model_name": model_name}

    monkeypatch.setitem(
        sys.modules, "langchain_google_vertexai", SimpleNamespace(VertexAIEmbeddings=DummyVertexEmbeddings)
    )
    _install_stub(monkeypatch, "rag_bencher.providers.gcp.embeddings.is_installed")

    adapter = VertexEmbeddingsAdapter(
        {"model": "text-embedding-004", "location": "asia-northeast1", "project_id": "proj"}
    )
    emb = adapter.to_langchain()

    assert isinstance(emb, DummyVertexEmbeddings)
    assert calls["kwargs"] == {
        "location": "asia-northeast1",
        "project": "proj",
        "model_name": "text-embedding-004",
    }


def test_vertex_embeddings_adapter_supports_model_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    class DummyVertexEmbeddings:
        def __init__(self, *, location: str, project: str | None, model: str) -> None:
            calls["kwargs"] = {"location": location, "project": project, "model": model}

    monkeypatch.setitem(
        sys.modules, "langchain_google_vertexai", SimpleNamespace(VertexAIEmbeddings=DummyVertexEmbeddings)
    )
    _install_stub(monkeypatch, "rag_bencher.providers.gcp.embeddings.is_installed")

    adapter = VertexEmbeddingsAdapter({"project_id": None})
    emb = adapter.to_langchain()

    assert isinstance(emb, DummyVertexEmbeddings)
    assert calls["kwargs"] == {"location": "us-central1", "project": None, "model": "text-embedding-004"}


def test_vertex_chat_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.gcp.chat.is_installed")
    adapter = VertexChatAdapter({})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[gcp]"):
        adapter.to_langchain()


def test_vertex_embeddings_adapter_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_install(monkeypatch, "rag_bencher.providers.gcp.embeddings.is_installed")
    adapter = VertexEmbeddingsAdapter({})
    with pytest.raises(RuntimeError, match="Install: rag-bencher\\[gcp]"):
        adapter.to_langchain()


def test_build_chat_adapter_dispatches_known_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(
        base.build_chat_adapter({"name": "azure", "chat": {"endpoint": "https://example"}}), AzureOpenAIChatAdapter
    )
    assert isinstance(base.build_chat_adapter({"name": "gcp", "chat": {}}), VertexChatAdapter)
    assert isinstance(base.build_chat_adapter({"name": "aws", "chat": {}, "region": "us-east-1"}), BedrockChatAdapter)
    assert base.build_chat_adapter(None) is None
    assert base.build_chat_adapter({}) is None
    with pytest.raises(ValueError, match="Unknown provider"):
        base.build_chat_adapter({"name": "unknown"})


def test_build_embeddings_adapter_dispatches_known_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(
        base.build_embeddings_adapter({"name": "azure", "embeddings": {"endpoint": "https://example"}}),
        AzureOpenAIEmbeddingsAdapter,
    )
    assert isinstance(base.build_embeddings_adapter({"name": "gcp", "embeddings": {}}), VertexEmbeddingsAdapter)
    assert isinstance(
        base.build_embeddings_adapter({"name": "aws", "embeddings": {}, "region": "us-east-1"}),
        BedrockEmbeddingsAdapter,
    )
    assert base.build_embeddings_adapter(None) is None
    assert base.build_embeddings_adapter({}) is None
    with pytest.raises(ValueError, match="Unknown provider"):
        base.build_embeddings_adapter({"name": "weird"})
