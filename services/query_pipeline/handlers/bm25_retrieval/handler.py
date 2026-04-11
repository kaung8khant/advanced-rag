from __future__ import annotations

from services.config import CHUNK_STORE_DIR, RETRIEVAL_TOP_K
from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.bm25_retrieval.methods import (
    retrieve_bm25_matches,
)


class Bm25RetrievalHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        question = str(context["question"])
        context["bm25_matches"] = retrieve_bm25_matches(
            question,
            top_k=RETRIEVAL_TOP_K,
            chunk_store_dir=CHUNK_STORE_DIR,
            min_score=0.0,
        )
        return context

