from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from fastuuid import uuid4
from services.answer_generator import answer_with_ollama, build_context_from_matches
from services.bm25_service import retrieve_bm25_matches
from services.config import (
    CHUNK_STORE_DIR,
    RERANKING_MODEL,
    RETRIEVAL_TOP_K,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
)
from services.file_extractor import extract_and_enrich_segments
from services.retrieval_service import (
    store_chunks_json,
    store_vectors_and_attach_faiss_ids,
)
from services.reranker import rerank_matches
from services.token_chunker import build_chunks_from_segments
from services.vectorizer import chunks_to_vectors
from services.multi_query_retriever import retrieve_multi_query_matches


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
    return [
        {**item, "k": idx}
        for idx, item in enumerate(ranked_matches, start=1)
    ]


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_STORE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


async def save_file(file: UploadFile) -> dict[str, str | bytes]:
    filename = file.filename or ""
    ext = f".{filename.split('.')[-1].lower()}"
    if ext not in {".pdf", ".md", ".markdown"}:
        raise HTTPException(
            status_code=400,
            detail="Only .pdf, .md, and .markdown files are allowed.",
        )

    safe_name = f"{uuid4().hex}{ext}"
    destination = UPLOAD_DIR / safe_name
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    destination.write_bytes(content)

    return {"filename": filename, "saved_as": safe_name, "content": content, "ext": ext}


async def upload_document(
    file: UploadFile,
    *,
    owner_id: str = "1",
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    file_info = await save_file(file)
    print(f"Saved file: {file_info['filename']} as {file_info['saved_as']}")

    extracted_value = extract_and_enrich_segments(
        content=file_info["content"],
        ext=file_info["ext"],
        saved_as=file_info["saved_as"],
        owner_id=owner_id,
    )
    chunks = build_chunks_from_segments(
        extracted_value, chunk_size=300, token_overlap=50
    )
    vectorized_chunks = chunks_to_vectors(chunks)
    chunks_with_faiss_ids = store_vectors_and_attach_faiss_ids(
        vectorized_chunks, vector_store_dir=VECTOR_STORE_DIR
    )
    doc_id = Path(str(file_info["saved_as"])).stem
    json_path = store_chunks_json(
        chunks_with_faiss_ids, doc_id=doc_id, chunk_store_dir=CHUNK_STORE_DIR
    )
    print(f"Saved chunk JSON: {json_path}")

    response_chunks = [
        {k: v for k, v in c.items() if k != "vector"} for c in chunks_with_faiss_ids
    ]
    message = f"File uploaded successfully.  {file_info['filename']} as {file_info['saved_as']}"
    return message, extracted_value, response_chunks


def ask_question(question: str) -> tuple[int, str, list[dict[str, Any]]]:
    cleaned = question.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="question must be non-empty")

    multi_query_matches = retrieve_multi_query_matches(cleaned, num_queries=5)
    # rewritten = rewrite_query_with_ollama(cleaned)
    # print(f"Rewritten query: {rewritten}")

    bm25_matches = retrieve_bm25_matches(
        cleaned,
        top_k=RETRIEVAL_TOP_K,
        chunk_store_dir=CHUNK_STORE_DIR,
        min_score=0.0,
    )

    merged_matches = merge_retrieval_matches(multi_query_matches, bm25_matches)
    final_matches = rerank_matches(
        cleaned,
        merged_matches,
        model=RERANKING_MODEL,
        top_k=RETRIEVAL_TOP_K,
    )
    if not final_matches:
        return RETRIEVAL_TOP_K, "No relevant context was found.", []

    context = build_context_from_matches(final_matches)
    answer = answer_with_ollama(cleaned, context=context)
    return RETRIEVAL_TOP_K, answer, final_matches
