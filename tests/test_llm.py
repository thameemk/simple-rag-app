from unittest.mock import patch

import llm


def test_build_prompt_includes_context_and_question():
    prompt = llm.build_prompt("some context", "some question")

    assert "some context" in prompt
    assert "some question" in prompt


@patch("llm.client")
def test_generate_answer_calls_client_complete(mock_client):
    mock_client.complete.return_value = "the answer"

    result = llm.generate_answer("some context", "some question")

    assert result == "the answer"
    mock_client.complete.assert_called_once_with(
        [{"role": "user", "content": llm.build_prompt("some context", "some question")}]
    )
