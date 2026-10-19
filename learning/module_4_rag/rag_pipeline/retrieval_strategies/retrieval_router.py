
import re

from rag_pipeline.models import RetrievalPlan


class RetrievalRouter:

    def route(self, query: str) -> RetrievalPlan:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_lower = query.lower()

        # Exact policy identifiers (illustrative format).
        has_policy_id = bool(
            re.search(
                r"\b[A-Z]{2}-\d{4}-[A-Z]{2}-\d+\b",
                query,
                flags=re.IGNORECASE
            )
        )

        if has_policy_id:
            return RetrievalPlan(
                strategy="keyword",
                use_reranker=False,
                use_mmr=False,
                reason="Exact policy identifier detected."
            )

        # Broad questions may benefit from diverse evidence.
        broad_terms = [
            "summarize",
            "overview",
            "explain the entire",
            "all policies",
            "compare",
        ]

        if any(term in query_lower for term in broad_terms):
            return RetrievalPlan(
                strategy="hybrid",
                use_reranker=True,
                use_mmr=True,
                reason="Broad or multi-aspect question."
            )

        # Simple factual questions.
        factual_starters = (
            "how many",
            "when is",
            "what is",
            "who is",
        )

        if query_lower.startswith(factual_starters):
            return RetrievalPlan(
                strategy="semantic",
                use_reranker=False,
                use_mmr=False,
                reason="Simple factual question."
            )

        # Default: robust retrieval from both systems.
        return RetrievalPlan(
            strategy="hybrid",
            use_reranker=True,
            use_mmr=False,
            reason="General question; hybrid fallback."
        )
