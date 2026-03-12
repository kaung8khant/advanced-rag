from __future__ import annotations

from typing import Any

import numpy as np
from clients.ollama_client import create_ollama_client


def text_to_vector(
    text: str,
    *,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[float]:
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    client = create_ollama_client(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def chunks_to_vectors(
    chunks: list[dict[str, Any]],
    *,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[dict[str, Any]]:
    client = create_ollama_client(base_url=base_url, api_key=api_key)
    results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            print(f"[{idx}] skipped: empty text")
            continue
        response = client.embeddings.create(model=model, input=text)
        vector = response.data[0].embedding
        chunk_id = chunk.get("chunk_id", f"chunk-{idx}")
        norm_value = float(np.linalg.norm(vector))
        print(
            f"[{idx}] embedded chunk_id={chunk_id} dim={len(vector)} norm={norm_value:.6f} vector_sample={vector[:5]}..."
        )
        results.append(
            {
                **chunk,
                "vector": vector,
                "vector_dim": len(vector),
            }
        )

    return results
