
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.context_builder import ContextBuilder


def main():
    embedding_service = EmbeddingService()
    vector_store = ChromaStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    context_builder = ContextBuilder(
        max_context_tokens=2000
    )

    query = (
        "How many annual leave days do employees receive, "
        "and can unused leave be carried forward?"
    )

    # Step 1: Retrieve relevant chunks
    results = retrieval_service.retrieve(
        query=query,
        top_k=5,
        similarity_threshold=0.30,
       filters={"document_id": 'handbook_001'}
    )

    # Step 2: Assemble LLM-ready context
    assembled = context_builder.build(results)

    print("\n===== RETRIEVED CONTEXT =====")
    print(assembled.text or "No context available.")

    print("\nEstimated tokens:", assembled.estimated_tokens)

    print("\n===== SOURCE REFERENCES =====")
    for source in assembled.sources:
        print(
            source.label,
            source.filename,
            f"Page {source.page_number}"
        )


if __name__ == "__main__":
    main()
