
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService


def main():
    embedding_service = EmbeddingService()
    vector_store = ChromaStore()

    if vector_store.count() == 0:
        print("No documents indexed. Run run_ingestion.py first.")
        return

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    print("Stored chunks:", vector_store.count())

    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieval_service.retrieve(
            query=query,
            top_k=5,
            similarity_threshold=0.20,
            filters={"document_id": "handbook_001"}
        )

        if not results:
            print("No sufficiently relevant chunks found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print("Chunk ID:", result.chunk_id)
            print("Document ID:", result.document_id)
            print("Source:", result.source)
            print("Page:", result.page_number)
            print("Similarity:", round(result.similarity, 4))
            print("Text:", result.text)


if __name__ == "__main__":
    main()
