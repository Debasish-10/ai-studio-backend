
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService

from rag_pipeline.retrieval_strategies.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval_strategies.rank_fusion import ReciprocalRankFusion
from rag_pipeline.retrieval_strategies.reranker import CrossEncoderReranker
from rag_pipeline.retrieval_strategies.mmr_selector import MMRSelector
from rag_pipeline.retrieval_strategies.query_processor import QueryProcessor
from rag_pipeline.retrieval_strategies.retrieval_router import RetrievalRouter
from rag_pipeline.retrieval_strategies.adaptive_retrieval import (
    AdaptiveRetrievalService
)


def main():
    filters = {"document_id": "handbook_001"}

    store = ChromaStore()
    chunks = store.get_chunks(filters=filters)

    if not chunks:
        print("No indexed chunks found.")
        return

    embeddings = EmbeddingService()

    # Initialize dependencies once.
    service = AdaptiveRetrievalService(
        dense=RetrievalService(
            embedding_service=embeddings,
            vector_store=store
        ),
        bm25=BM25Retriever(chunks),
        fusion=ReciprocalRankFusion(),
        reranker=CrossEncoderReranker(),
        mmr=MMRSelector(
            embedding_service=embeddings,
            lambda_param=0.5
        ),
        router=RetrievalRouter()
    )

    processor = QueryProcessor()

    while True:
        query = input("\nQuestion (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        rewritten = processor.rewrite(query)

        plan, results = service.retrieve(
            query=rewritten,
            filters=filters,
            top_k=5
        )

        print("\nOriginal:", query)
        print("Search query:", rewritten)
        print("Strategy:", plan.strategy)
        print("Reranker:", plan.use_reranker)
        print("MMR:", plan.use_mmr)
        print("Reason:", plan.reason)

        print("\n===== RESULTS =====")

        for i, result in enumerate(results, 1):
            print(
                i,
                result.chunk_id,
                result.text[:150]
            )


if __name__ == "__main__":
    main()
