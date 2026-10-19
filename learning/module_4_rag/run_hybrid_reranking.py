
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService

from rag_pipeline.retrieval_strategies.bm25_retriever import (
    BM25Retriever
)
from rag_pipeline.retrieval_strategies.rank_fusion import (
    ReciprocalRankFusion
)
from rag_pipeline.retrieval_strategies.reranker import (
    CrossEncoderReranker
)

from rag_pipeline.retrieval_strategies.mmr_selector import (
    MMRSelector
)


def main():
    store = ChromaStore()

    filters = {"document_id": "handbook_001"}

    chunks = store.get_chunks(filters=filters)

    if not chunks:
        print("No indexed chunks found.")
        return

    bm25 = BM25Retriever(chunks)

    embedding_service = EmbeddingService()

    dense = RetrievalService(
        embedding_service=embedding_service,
        vector_store=store
    )

    mmr_selector = MMRSelector(
        embedding_service=embedding_service,
        lambda_param=0.5
    )

    fusion = ReciprocalRankFusion(rank_constant=60)

    # Load only once, not for every question.
    reranker = CrossEncoderReranker()

    while True:
        query = input("\nQuestion (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # 1. Semantic retrieval
        dense_results = dense.retrieve(
            query=query,
            top_k=10,
            similarity_threshold=0.0,
            filters=filters
        )

        # 2. Keyword retrieval
        bm25_results = bm25.search(
            query=query,
            top_k=10
        )

        # 3. Combine rankings
        fused_results = fusion.fuse(
            [dense_results, bm25_results],
            top_k=15
        )
        # Keep a larger pool after reranking.
        reranked_candidates = reranker.rerank(
            query=query,
            candidates=fused_results,
            top_k=15
        )

        # Select a diverse final set.
        final_results = mmr_selector.select(
            candidates=reranked_candidates,
            top_k=5
        )

        print("\n===== RRF FUSED RESULTS =====")

        for index, result in enumerate(fused_results, 1):
            print(
                index,
                result.chunk_id,
                round(result.score, 5)
            )

            print("\n===== CROSS-ENCODER TOP-5 =====")

        for item in reranked_candidates[:5]:
            print(item.chunk_id, item.text[:100])


        print("\n===== MMR SELECTED TOP-5 =====")

        for item in final_results:
            print(item.chunk_id, item.text[:100])

if __name__ == "__main__":
    main()
