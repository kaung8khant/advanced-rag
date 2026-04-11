from __future__ import annotations

from services.query_pipeline.base import QueryHandler
from services.query_pipeline.handlers.answer_generation import AnswerGenerationHandler
from services.query_pipeline.handlers.bm25_retrieval import Bm25RetrievalHandler
from services.query_pipeline.handlers.merge_matches import MergeMatchesHandler
from services.query_pipeline.handlers.multi_query_retrieval import (
    MultiQueryRetrievalHandler,
)
from services.query_pipeline.handlers.query_preparation import (
    QueryPreparationHandler,
)
from services.query_pipeline.handlers.rerank_matches import RerankMatchesHandler


def build_query_pipeline() -> QueryHandler:
    preparation = QueryPreparationHandler()
    (
        preparation.set_next_handler(MultiQueryRetrievalHandler())
        .set_next_handler(Bm25RetrievalHandler())
        .set_next_handler(MergeMatchesHandler())
        .set_next_handler(RerankMatchesHandler())
        .set_next_handler(AnswerGenerationHandler())
    )
    return preparation
