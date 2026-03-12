from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ExtractTextResponseDTO(BaseModel):
    message: str
    extracted_text: list[dict[str, Any]]
    token_chunks: list[dict[str, Any]]


class AskRequestDTO(BaseModel):
    question: str


class AskResponseDTO(BaseModel):
    question: str
    top_k: int
    answer: str
    matches: list[dict[str, Any]]
