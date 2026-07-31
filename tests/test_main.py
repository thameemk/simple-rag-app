from unittest.mock import patch

import main
from fastapi.testclient import TestClient

client = TestClient(main.app)


@patch("llm.client")
def test_chat_endpoint_returns_answer(mock_chat_client):
    mock_chat_client.complete.return_value = "mocked answer"

    response = client.post("/chat", json={"question": "What is this app about?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "mocked answer"}
    mock_chat_client.complete.assert_called_once()


def test_chat_endpoint_requires_question_field():
    response = client.post("/chat", json={})

    assert response.status_code == 422
