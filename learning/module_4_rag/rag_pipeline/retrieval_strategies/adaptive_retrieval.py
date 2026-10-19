
from rag_pipeline.models import RankedCandidate


def to_ranked(results):
    return [
        RankedCandidate(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            text=r.text,
            source=r.source,
            page_number=r.page_number,
            score=r.score if hasattr(r, "score") else r.similarity
        )
        for r in results
    ]


class AdaptiveRetrievalService:

    def __init__(
        self,
        dense,
        bm25,
        fusion,
        reranker,
        mmr,
        router
    ):
        self.dense = dense
        self.bm25 = bm25
        self.fusion = fusion
        self.reranker = reranker
        self.mmr = mmr
        self.router = router

    def retrieve(
        self,
        query: str,
        filters: dict,
        top_k: int = 5
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        plan = self.router.route(query)
        candidate_k = max(20, top_k)

        if plan.strategy == "semantic":
            candidates = to_ranked(
                self.dense.retrieve(
                    query=query,
                    top_k=candidate_k,
                    similarity_threshold=0.0,
                    filters=filters
                )
            )

        elif plan.strategy == "keyword":
            candidates = to_ranked(
                self.bm25.search(
                    query=query,
                    top_k=candidate_k
                )
            )

        else:
            dense_results = self.dense.retrieve(
                query=query,
                top_k=candidate_k,
                similarity_threshold=0.0,
                filters=filters
            )

            bm25_results = self.bm25.search(
                query=query,
                top_k=candidate_k
            )

            candidates = self.fusion.fuse(
                [dense_results, bm25_results],
                top_k=candidate_k
            )

        if plan.use_reranker and candidates:
            candidates = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=candidate_k
            )

        if plan.use_mmr and candidates:
            candidates = self.mmr.select(
                candidates=candidates,
                top_k=top_k
            )
        else:
            candidates = candidates[:top_k]

        return plan, candidates
