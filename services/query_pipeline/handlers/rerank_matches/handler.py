from __future__ import annotations

from services.config import RERANKING_MODEL, RETRIEVAL_TOP_K
from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.rerank_matches.methods import rerank_matches


class RerankMatchesHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        question = str(context["question"])
        merged_matches = list(context.get("merged_matches", []))
        context["final_matches"] = rerank_matches(
            question,
            merged_matches,
            model=RERANKING_MODEL,
            top_k=RETRIEVAL_TOP_K,
        )
        return context

