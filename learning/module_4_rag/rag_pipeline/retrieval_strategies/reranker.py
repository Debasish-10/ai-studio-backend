
from sentence_transformers import CrossEncoder

from rag_pipeline.models import RankedCandidate


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RankedCandidate],
        top_k: int = 5
    ) -> list[RankedCandidate]:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not candidates:
            return []

        pairs = [
            (query, candidate.text)
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = [
            RankedCandidate(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                text=candidate.text,
                source=candidate.source,
                page_number=candidate.page_number,
                score=float(score)
            )
            for candidate, score in zip(candidates, scores)
        ]

        reranked.sort(
            key=lambda item: item.score,
            reverse=True
        )

        return reranked[:top_k]
