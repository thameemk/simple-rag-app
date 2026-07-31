import pytest
from clients.chat import get_chat_client
from integrations.huggingface_client import HuggingFaceChatClient
from integrations.openai_client import OpenAIChatClient


def test_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = get_chat_client()

    assert isinstance(client, OpenAIChatClient)


def test_selects_huggingface(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    client = get_chat_client()

    assert isinstance(client, HuggingFaceChatClient)


def test_provider_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "HuggingFace")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    client = get_chat_client()

    assert isinstance(client, HuggingFaceChatClient)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_chat_client()
