from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from fastuuid import uuid4
from services.answer_generator import answer_with_ollama, build_context_from_matches
from services.file_extractor import extract_and_enrich_segments
from services.query_rewriter import rewrite_query_with_ollama
from services.retrieval_service import (
    retrieve_ranked_matches,
    store_chunks_json,
    store_vectors_and_attach_faiss_ids,
)
from services.token_chunker import build_chunks_from_segments
from services.vectorizer import chunks_to_vectors, text_to_vector

UPLOAD_DIR = Path("raw")
CHUNK_STORE_DIR = Path("chunk_store")
VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL = "nomic-embed-text"
ANSWER_MODEL = "llama3.2:latest"
RETRIEVAL_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.55


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
    message = (
        f"File uploaded successfully.  {file_info['filename']} as {file_info['saved_as']}"
    )
    return message, extracted_value, response_chunks


def ask_question(question: str) -> tuple[int, str, list[dict[str, Any]]]:
    cleaned = question.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="question must be non-empty")

    rewritten = rewrite_query_with_ollama(cleaned)
    print(rewritten)
    vector = text_to_vector(rewritten, model=EMBEDDING_MODEL)
    ranked_matches = retrieve_ranked_matches(
        vector,
        top_k=RETRIEVAL_TOP_K,
        min_score=MIN_RETRIEVAL_SCORE,
        vector_store_dir=VECTOR_STORE_DIR,
        chunk_store_dir=CHUNK_STORE_DIR,
    )
    context = build_context_from_matches(ranked_matches)
    answer = answer_with_ollama(cleaned, context=context, model=ANSWER_MODEL)
    return RETRIEVAL_TOP_K, answer, ranked_matches
