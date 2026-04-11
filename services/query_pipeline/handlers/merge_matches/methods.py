from __future__ import annotations

from typing import Any


def merge_retrieval_matches(
    *match_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_matches: dict[str, dict[str, Any]] = {}

    for match_group in match_groups:
        for match in match_group:
            chunk = match.get("chunk", {})
            faiss_id = match.get("faiss_id")
            key = str(faiss_id) if faiss_id is not None else ""
            if not key:
                key = str(chunk.get("chunk_id", "")).strip()
            if not key:
                key = str(chunk.get("text", "")).strip()
            if not key:
                continue

            retrieval_method = str(match.get("retrieval_method", "")).strip()
            existing = merged_matches.get(key)
            if existing is None:
                merged_matches[key] = {
                    **match,
                    "retrieval_method": retrieval_method or "unknown",
                }
                continue

            if float(match.get("score", 0.0)) > float(existing.get("score", 0.0)):
                existing["score"] = match["score"]
                existing["chunk"] = chunk

            methods = {
                item
                for item in [
                    str(existing.get("retrieval_method", "")).strip(),
                    retrieval_method,
                ]
                if item
            }
            existing["retrieval_method"] = "+".join(sorted(methods))

            existing_queries = existing.get("matched_queries")
            incoming_queries = match.get("matched_queries")
            if isinstance(existing_queries, list) and isinstance(incoming_queries, list):
                for query in incoming_queries:
                    if query not in existing_queries:
                        existing_queries.append(query)

    ranked_matches = sorted(
        merged_matches.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    return [{**item, "k": idx} for idx, item in enumerate(ranked_matches, start=1)]

