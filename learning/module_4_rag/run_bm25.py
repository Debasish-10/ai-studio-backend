
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.retrieval_strategies.bm25_retriever import (
    BM25Retriever
)


def main():
    store = ChromaStore()

    filters = {"document_id": "handbook_001"}

    chunks = store.get_chunks(filters=filters)

    if not chunks:
        print("No matching chunks found. Run ingestion first.")
        return

    bm25 = BM25Retriever(chunks)

    dense = RetrievalService(
        embedding_service=EmbeddingService(),
        vector_store=store
    )

    while True:
        query = input("\nQuestion (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        dense_results = dense.retrieve(
            query=query,
            top_k=5,
            similarity_threshold=0.0,
            filters=filters
        )

        bm25_results = bm25.search(
            query=query,
            top_k=5
        )

        print("\n===== SEMANTIC SEARCH =====")
        for result in dense_results:
            print(
                f"{result.similarity:.3f}",
                result.chunk_id,
                result.text[:120]
            )

        print("\n===== BM25 SEARCH =====")
        for result in bm25_results:
            print(
                f"{result.score:.3f}",
                result.chunk_id,
                result.text[:120]
            )


if __name__ == "__main__":
    main()
