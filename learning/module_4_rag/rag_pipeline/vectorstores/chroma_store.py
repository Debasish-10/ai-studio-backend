import chromadb
import numpy as np

from rag_pipeline.config import (
    CHROMA_PATH,
    COLLECTION_NAME
)

from rag_pipeline.models import (
    DocumentChunk,
    RetrievalResult
)


class ChromaStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def replace_document(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray
    ) -> None:

        if not chunks:
            raise ValueError("Cannot store empty chunks.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings count must match."
            )

        ids = [chunk.id for chunk in chunks]

        documents = [chunk.text for chunk in chunks]

        metadatas = [
            {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page_number,
                "chunk_number": chunk.chunk_number
            }
            for chunk in chunks
        ]

        # Remove old chunks belonging to this document.
        self.collection.delete(
            where={"document_id": document_id}
        )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def count(self) -> int:
        return self.collection.count()

    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        # Chroma expects a batch of query embeddings.
        # Our EmbeddingService returns shape (1, 384).
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            metadata = metadata or {}

            matches.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=metadata["document_id"],
                    text=text,
                    source=metadata["source"],
                    page_number=metadata["page"],
                    similarity=1.0 - float(distance)
                )
            )

        return matches


    def get_chunks(
    self,
    filters: dict | None = None,
    batch_size: int = 100
) -> list[dict]:

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        chunks = []
        offset = 0

        while True:
            batch = self.collection.get(
                where=filters,
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )

            if not batch["ids"]:
                break

            for chunk_id, text, metadata in zip(
                batch["ids"],
                batch["documents"],
                batch["metadatas"]
            ):
                chunks.append({
                    "id": chunk_id,
                    "document_id": metadata["document_id"],
                    "text": text,
                    "source": metadata["source"],
                    "page": metadata["page"]
                })

            offset += len(batch["ids"])

            if len(batch["ids"]) < batch_size:
                break

        return chunks
