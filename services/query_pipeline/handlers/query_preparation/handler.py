from __future__ import annotations

from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.query_preparation.methods import (
    prepare_query_context,
)


class QueryPreparationHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        return prepare_query_context(context)

