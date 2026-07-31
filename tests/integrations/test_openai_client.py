from unittest.mock import patch

from integrations.openai_client import OpenAIChatClient


@patch("integrations.openai_client.OpenAI")
def test_complete_returns_stripped_content(mock_openai_cls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_create = mock_openai_cls.return_value.chat.completions.create
    mock_create.return_value.choices[0].message.content = "  hello world  "

    client = OpenAIChatClient()
    result = client.complete([{"role": "user", "content": "hi"}])

    assert result == "hello world"
    mock_create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_uses_model_override_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAIChatClient()

    assert client._model == "gpt-4o"
