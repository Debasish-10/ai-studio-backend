
from dotenv import load_dotenv

from rag_pipeline.embeddings.embedding_service import (
    EmbeddingService
)
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.context_builder import ContextBuilder

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
from rag_pipeline.retrieval_strategies.retrieval_router import (
    RetrievalRouter
)
from rag_pipeline.retrieval_strategies.adaptive_retrieval import (
    AdaptiveRetrievalService
)

from rag_pipeline.generation.prompt_builder import (
    RAGPromptBuilder
)
from rag_pipeline.generation.groq_llm import (
    GroqLLMService,
    LLMGenerationError
)
from rag_pipeline.generation.rag_generation import (
    RAGGenerationService
)
from rag_pipeline.generation.citation_validator import (
    CitationValidator
)


def main():
    load_dotenv()

    # Learning scope: one indexed handbook.
    # Production scope must come from backend authorization.
    filters = {"document_id": "handbook_001"}

    vector_store = ChromaStore()
    chunks = vector_store.get_chunks(filters=filters)

    if not chunks:
        print("No chunks available. Run ingestion first.")
        return

    # Initialize reusable models and dependencies.
    embeddings = EmbeddingService()

    dense = RetrievalService(
        embedding_service=embeddings,
        vector_store=vector_store
    )

    adaptive = AdaptiveRetrievalService(
        dense=dense,
        bm25=BM25Retriever(chunks),
        fusion=ReciprocalRankFusion(),
        reranker=CrossEncoderReranker(),
        mmr=MMRSelector(
            embedding_service=embeddings,
            lambda_param=0.5
        ),
        router=RetrievalRouter()
    )

 
    rag = RAGGenerationService(
        retrieval_service=adaptive,
        context_builder=ContextBuilder(
            max_context_tokens=2000
        ),
        prompt_builder=RAGPromptBuilder(),
        llm_service=GroqLLMService(),
        citation_validator=CitationValidator()
    )

    print("Grounded RAG assistant ready.")

    while True:
        query = input("\nAsk a question ('exit' to quit): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:
            result = rag.generate(
                query=query,
                filters=filters,
                top_k=5
            )

            print("\nSTATUS:", result.status)
            print("\nANSWER:\n", result.answer)

            print("\nAVAILABLE SOURCES:")
            for source in result.sources:
                print(
                    source.label,
                    source.filename,
                    f"Page {source.page_number}",
                    source.chunk_id
                )

            print("\nMODEL:", result.model)
            print("INPUT TOKENS:", result.input_tokens)
            print("OUTPUT TOKENS:", result.output_tokens)

        except LLMGenerationError as exc:
            print("Generation failed:", exc)

        except Exception as exc:
            # Learning-only diagnostic. In production,
            # classify errors and use structured logging.
            print("Pipeline error:", type(exc).__name__, exc)

def create_rag_service() -> RAGGenerationService:
    load_dotenv()

    filters = {"document_id": "handbook_001"}

    vector_store = ChromaStore()
    chunks = vector_store.get_chunks(filters=filters)

    if not chunks:
        raise RuntimeError("No chunks available. Run ingestion first.")

    embeddings = EmbeddingService()

    dense = RetrievalService(
        embedding_service=embeddings,
        vector_store=vector_store
    )

    adaptive = AdaptiveRetrievalService(
        dense=dense,
        bm25=BM25Retriever(chunks),
        fusion=ReciprocalRankFusion(),
        reranker=CrossEncoderReranker(),
        mmr=MMRSelector(
            embedding_service=embeddings,
            lambda_param=0.5
        ),
        router=RetrievalRouter()
    )

    rag = RAGGenerationService(
        retrieval_service=adaptive,
        context_builder=ContextBuilder(
            max_context_tokens=2000
        ),
        prompt_builder=RAGPromptBuilder(),
        llm_service=GroqLLMService(),
        citation_validator=CitationValidator()
    )

    return rag

if __name__ == "__main__":
    main()
