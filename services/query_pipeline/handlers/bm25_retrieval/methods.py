from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_bm25_chunks(chunk_store_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not chunk_store_dir.exists():
        return chunks

    for path in sorted(chunk_store_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            chunks.append({k: v for k, v in item.items() if k != "vector"})

    return chunks


def retrieve_bm25_matches(
    query_text: str,
    *,
    top_k: int,
    chunk_store_dir: Path,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []

    chunks = load_bm25_chunks(chunk_store_dir)
    if not chunks:
        return []

    tokenized_corpus: list[list[str]] = []
    indexed_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        tokens = tokenize_for_bm25(str(chunk.get("text", "")))
        if not tokens:
            continue
        tokenized_corpus.append(tokens)
        indexed_chunks.append(chunk)

    if not tokenized_corpus:
        return []

    query_tokens = tokenize_for_bm25(cleaned_query)
    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    ranked_pairs = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    matches: list[dict[str, Any]] = []
    for index, score in ranked_pairs:
        numeric_score = float(score)
        if numeric_score < min_score:
            continue
        chunk = indexed_chunks[index]
        matches.append(
            {
                "faiss_id": chunk.get("faiss_id"),
                "score": numeric_score,
                "retrieval_method": "bm25",
                "chunk": chunk,
            }
        )
        if len(matches) >= top_k:
            break

    ranked: list[dict[str, Any]] = []
    for idx, item in enumerate(matches, start=1):
        ranked.append({"k": idx, **item})
    return ranked

