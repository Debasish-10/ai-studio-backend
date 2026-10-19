
from rag_pipeline.retrieval_strategies.retrieval_router import (
    RetrievalRouter
)


def test_policy_identifier_uses_keyword_search():
    router = RetrievalRouter()

    plan = router.route(
        "Explain policy HR-2026-LV-042"
    )

    assert plan.strategy == "keyword"
    assert plan.use_reranker is False


def test_broad_question_uses_mmr():
    router = RetrievalRouter()

    plan = router.route(
        "Summarize the employee leave policy"
    )

    assert plan.strategy == "hybrid"
    assert plan.use_reranker is True
    assert plan.use_mmr is True


def test_factual_question_uses_semantic_search():
    router = RetrievalRouter()

    plan = router.route(
        "How many annual leave days are available?"
    )

    assert plan.strategy == "semantic"
    assert plan.use_mmr is False
