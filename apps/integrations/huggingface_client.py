import os

from openai import OpenAI


class HuggingFaceChatClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
        self._model = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:novita")

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
