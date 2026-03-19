from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError

from clients.ollama_client import create_ollama_client
from services.config import (
    ANSWER_MODEL,
    CHUNK_STORE_DIR,
    EMBEDDING_MODEL,
    MIN_RETRIEVAL_SCORE,
    RETRIEVAL_TOP_K,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
)


def build_context_from_matches(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return ""

    lines: list[str] = []
    for item in matches:
        chunk = item.get("chunk", {})
        lines.append(
            f"[k={item.get('k')}, faiss_id={item.get('faiss_id')}, score={item.get('score')}]"
        )
        lines.append(str(chunk.get("text", "")))
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_ollama(
    question: str,
    *,
    context: str,
    model: str = ANSWER_MODEL,
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be non-empty")

    client = create_ollama_client(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided context. "
                        "If context is insufficient, say you do not have enough information."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{cleaned}\n\nContext:\n{context}",
                },
            ],
        )
    except (APIConnectionError, APIError):
        return "Answer generation is unavailable because Ollama is not reachable."

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        return "The model returned an empty answer."
    return answer
