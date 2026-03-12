from __future__ import annotations

from openai import OpenAI

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"


def create_ollama_client(
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    api_key: str = DEFAULT_OLLAMA_API_KEY,
) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
