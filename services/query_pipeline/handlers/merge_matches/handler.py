from __future__ import annotations

from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.merge_matches.methods import (
    merge_retrieval_matches,
)


class MergeMatchesHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        context["merged_matches"] = merge_retrieval_matches(
            context.get("multi_query_matches", []),
            context.get("bm25_matches", []),
        )
        return context

