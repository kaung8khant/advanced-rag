from __future__ import annotations

from services.query_pipeline.base import QueryHandler
from services.query_pipeline.context import QueryPipelineContext
from services.query_pipeline.handlers.answer_generation.methods import (
    answer_with_ollama,
    build_context_from_matches,
)


class AnswerGenerationHandler(QueryHandler):
    def process(self, context: QueryPipelineContext) -> QueryPipelineContext:
        final_matches = list(context.get("final_matches", []))
        if not final_matches:
            context["answer"] = "No relevant context was found."
            return context

        question = str(context["question"])
        context["context"] = build_context_from_matches(final_matches)
        context["answer"] = answer_with_ollama(question, context=context["context"])
        return context

