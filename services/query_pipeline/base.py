from __future__ import annotations

from services.query_pipeline.context import QueryPipelineContext


class QueryHandler:
    def __init__(self, next_handler: QueryHandler | None = None) -> None:
        self.next_handler = next_handler

    def set_next_handler(self, next_handler: QueryHandler) -> QueryHandler:
        self.next_handler = next_handler
        return next_handler

    def handle(self, context: QueryPipelineContext) -> QueryPipelineContext:
        updated_context = self.process(context)
        if self.next_handler is None:
            return updated_context
        return self.next_handler.handle(updated_context)

    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        raise NotImplementedError
