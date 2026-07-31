import os
from typing import Protocol

from dotenv import load_dotenv
from integrations.huggingface_client import HuggingFaceChatClient
from integrations.openai_client import OpenAIChatClient

load_dotenv()


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


_PROVIDERS = {
    "openai": OpenAIChatClient,
    "huggingface": HuggingFaceChatClient,
}


def get_chat_client() -> ChatClient:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        return _PROVIDERS[provider]()
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}. Choose from: {', '.join(_PROVIDERS)}"
        ) from None
