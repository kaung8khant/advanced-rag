from __future__ import annotations

from clients.ollama_client import create_ollama_client


def rewrite_query_with_ollama(
    query: str,
    *,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must be non-empty")

    client = create_ollama_client(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user query for semantic retrieval. "
                    "Keep original intent and key entities. "
                    "Return only one rewritten query with no explanation."
                    "Find keywords and entities in the query and rewrite it to be more effective for retrieval. Don't remove original keywords or entities, but feel free to add more. Don't say you are rewriting, just return the rewritten query."
                ),
            },
            {"role": "user", "content": cleaned},
        ],
    )

    rewritten = (response.choices[0].message.content or "").strip()
    if not rewritten:
        raise RuntimeError("Ollama returned an empty rewritten query")
    return rewritten
