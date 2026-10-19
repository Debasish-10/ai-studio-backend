
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService


def main():
    service = RetrievalService(
        embedding_service=EmbeddingService(),
        vector_store=ChromaStore()
    )

    query = "How many annual leave days do employees receive?"

    # Test 1: Correct document ID
    results = service.retrieve(
        query=query,
        top_k=5,
        similarity_threshold=0.30,
        filters={
            "document_id": "handbook_001"
        }
    )

    print("\nTEST 1: Valid document filter")

    for result in results:
        print(
            result.chunk_id,
            round(result.similarity, 3),
            result.text[:100]
        )

    # Test 2: Wrong document ID
    results = service.retrieve(
        query=query,
        top_k=5,
        similarity_threshold=0.30,
        filters={
            "document_id": "unknown_document"
        }
    )

    print("\nTEST 2: Invalid document ID")
    print("Results:", len(results))

    # Test 3: Multiple allowed document IDs
    results = service.retrieve(
        query=query,
        top_k=5,
        similarity_threshold=0.30,
        filters={
            "document_id": {
                "$in": [
                    "handbook_001",
                    "benefits_002"
                ]
            }
        }
    )

    print("\nTEST 3: Multiple document IDs")

    for result in results:
        print(
            result.document_id,
            result.chunk_id
        )


if __name__ == "__main__":
    main()
