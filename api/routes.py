from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from schemas.api_schemas import (
    AskRequestDTO,
    AskResponseDTO,
    ExtractTextResponseDTO,
)
from services.rag_service import ask_question, upload_document

router = APIRouter()

@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the RAG API! Use /upload to upload documents and /ask to ask questions."}

@router.post("/upload", response_model=ExtractTextResponseDTO)
async def upload_file(
    file: UploadFile = File(...), owner_id: str = "1"
) -> ExtractTextResponseDTO:
    message, extracted_text, token_chunks = await upload_document(file, owner_id=owner_id)
    return ExtractTextResponseDTO(
        message=message,
        extracted_text=extracted_text,
        token_chunks=token_chunks,
    )


@router.post("/ask", response_model=AskResponseDTO)
async def ask(payload: AskRequestDTO) -> AskResponseDTO:
    top_k, answer, matches = ask_question(payload.question)
    return AskResponseDTO(
        question=payload.question,
        top_k=top_k,
        answer=answer,
        matches=matches,
    )
