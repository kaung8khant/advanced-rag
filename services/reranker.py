from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank_matches(
    query: str,
    matches: list[dict[str, Any]],
    *,
    model: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    cleaned_query = query.strip()
    if not cleaned_query or not matches:
        return matches

    pairs: list[list[str]] = []
    valid_matches: list[dict[str, Any]] = []
    for match in matches:
        text = str(match.get("chunk", {}).get("text", "")).strip()
        if not text:
            continue
        pairs.append([cleaned_query, text])
        valid_matches.append(match)

    if not pairs:
        return matches

    reranker = _load_reranker(model)
    scores = reranker.predict(pairs)

    reranked_matches = [
        {**match, "rerank_score": float(score)}
        for match, score in zip(valid_matches, scores, strict=False)
    ]
    ranked = sorted(
        reranked_matches,
        key=lambda item: float(item.get("rerank_score", 0.0)),
        reverse=True,
    )
    return [
        {**item, "k": rank}
        for rank, item in enumerate(
            ranked if top_k is None else ranked[:top_k],
            start=1,
        )
    ]
