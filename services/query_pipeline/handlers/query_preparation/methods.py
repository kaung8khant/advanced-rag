from __future__ import annotations

from fastapi import HTTPException

from services.config import RETRIEVAL_TOP_K
from services.query_pipeline.context import QueryPipelineContext


def prepare_query_context(context: QueryPipelineContext) -> QueryPipelineContext:
    question = str(context.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must be non-empty")

    context["question"] = question
    context["top_k"] = RETRIEVAL_TOP_K
    return context

