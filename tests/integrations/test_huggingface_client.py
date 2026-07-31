from unittest.mock import patch

from integrations.huggingface_client import HuggingFaceChatClient


@patch("integrations.huggingface_client.OpenAI")
def test_complete_returns_stripped_content(mock_openai_cls, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    mock_create = mock_openai_cls.return_value.chat.completions.create
    mock_create.return_value.choices[0].message.content = "  hello world  "

    client = HuggingFaceChatClient()
    result = client.complete([{"role": "user", "content": "hi"}])

    assert result == "hello world"
    mock_create.assert_called_once_with(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_uses_model_override_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_MODEL", "some/other-model")
    client = HuggingFaceChatClient()

    assert client._model == "some/other-model"
