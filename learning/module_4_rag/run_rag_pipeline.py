
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.context_builder import ContextBuilder
from rag_pipeline.pipeline import RAGPipeline


def main():
    # Initialize shared resources once
    embedding_service = EmbeddingService()
    vector_store = ChromaStore()

    if vector_store.count() == 0:
        print("No documents indexed.")
        print("Run python run_ingestion.py first.")
        return

    # Assemble services
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    context_builder = ContextBuilder(
        max_context_tokens=2000
    )

    rag_pipeline = RAGPipeline(
        retrieval_service=retrieval_service,
        context_builder=context_builder
    )

    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        result = rag_pipeline.prepare_context(
            query=query,
            document_id="handbook_001",
            top_k=5,
            similarity_threshold=0.50
        )

        print("\nStatus:", result.status)
        print("Retrieved chunks:", result.retrieved_count)

        if result.status == "no_context":
            print("No suitable document context found.")
            continue

        print("\n===== LLM READY CONTEXT =====")
        print(result.context.text)

        print(
            "\nEstimated tokens:",
            result.context.estimated_tokens
        )

        print("\n===== CITATIONS =====")

        for source in result.context.sources:
            print(
                source.label,
                source.filename,
                f"Page {source.page_number}"
            )


if __name__ == "__main__":
    main()
