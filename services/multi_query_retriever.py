from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clients.ollama_client import create_ollama_client
from services.config import (
    CHUNK_STORE_DIR,
    EMBEDDING_MODEL,
    MIN_RETRIEVAL_SCORE,
    MULTI_QUERY_COUNT,
    RETRIEVAL_TOP_K,
    VECTOR_STORE_DIR,
)
from services.retrieval_service import retrieve_ranked_matches
from services.vectorizer import text_to_vector


def generate_multi_queries_with_ollama(
    question: str,
    *,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    num_queries: int = 3,
) -> list[str]:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be non-empty")

    client = create_ollama_client(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate alternative search queries for retrieval. "
                    f"Return exactly {num_queries} queries as a JSON array of strings. "
                    "Keep entities and intent unchanged. No extra text."
                ),
            },
            {"role": "user", "content": cleaned},
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return [cleaned]

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            variants = [str(item).strip() for item in parsed if str(item).strip()]
            if variants:
                return variants[:num_queries]
    except json.JSONDecodeError:
        pass

    return [cleaned]


def retrieve_multi_query_matches(
    question: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = MIN_RETRIEVAL_SCORE,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    chunk_store_dir: Path = CHUNK_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    num_queries: int = MULTI_QUERY_COUNT,
    per_query_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Run multi-query semantic retrieval and return one merged ranked result list."""
    cleaned = question.strip()
    if not cleaned:
        return []

    generated_queries = generate_multi_queries_with_ollama(
        cleaned, num_queries=num_queries
    )
    all_queries = [cleaned, *generated_queries]

    seen_queries: set[str] = set()
    unique_queries: list[str] = []
    for query in all_queries:
        normalized = query.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen_queries:
            continue
        seen_queries.add(lowered)
        unique_queries.append(normalized)

    each_top_k = per_query_top_k or top_k
    merged: dict[int, dict[str, Any]] = {}
    for query in unique_queries:
        vector = text_to_vector(query, model=embedding_model)
        query_matches = retrieve_ranked_matches(
            vector,
            top_k=each_top_k,
            min_score=min_score,
            vector_store_dir=vector_store_dir,
            chunk_store_dir=chunk_store_dir,
        )

        for item in query_matches:
            faiss_id = int(item["faiss_id"])
            score = float(item["score"])
            existing = merged.get(faiss_id)
            if existing is None:
                merged[faiss_id] = {
                    "faiss_id": faiss_id,
                    "score": score,
                    "chunk": item["chunk"],
                    "matched_queries": [query],
                    "retrieval_method": "multi_query_faiss",
                }
                continue

            if score > float(existing["score"]):
                existing["score"] = score
                existing["chunk"] = item["chunk"]
            if query not in existing["matched_queries"]:
                existing["matched_queries"].append(query)

    ranked = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
    output: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked[:top_k], start=1):
        output.append({"k": idx, **item})
    return output
