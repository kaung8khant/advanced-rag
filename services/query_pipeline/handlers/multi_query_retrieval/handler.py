from __future__ import annotations

from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.multi_query_retrieval.methods import (
    retrieve_multi_query_matches,
)


class MultiQueryRetrievalHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        question = str(context["question"])
        context["multi_query_matches"] = retrieve_multi_query_matches(
            question,
            num_queries=5,
        )
        return context

