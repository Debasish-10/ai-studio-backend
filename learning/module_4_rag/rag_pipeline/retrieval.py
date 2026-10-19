
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.models import RetrievalResult


class RetrievalService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

   
    def retrieve(
    self,
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.50,
    filters: dict | None = None
) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("Invalid cosine similarity threshold.")

        query_embedding = self.embedding_service.embed_query(
            query
        )

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        return [
            item
            for item in candidates
            if item.similarity >= similarity_threshold
    ]