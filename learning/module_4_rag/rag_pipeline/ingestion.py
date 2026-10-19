from pathlib import Path

from rag_pipeline.loaders.pdf_loader import PDFLoader
from rag_pipeline.chunking.text_chunker import TextChunker
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore


class IngestionService:

    def __init__(
        self,
        loader: PDFLoader,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest(
        self,
        file_path: Path,
        document_id: str
    ) -> dict:

        # Step 1: Extract text
        pages = self.loader.load(file_path)

        if not pages:
            raise ValueError(
                "No extractable text found in PDF."
            )

        # Step 2: Create chunks
        chunks = self.chunker.split(
            pages,
            document_id
        )

        if not chunks:
            raise ValueError(
                "No chunks generated from document."
            )

        # Step 3: Generate embeddings
        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_service.embed_documents(
            texts
        )

        # Step 4: Persist chunks and embeddings
        self.vector_store.replace_document(
            document_id,
            chunks,
            embeddings
        )

        return {
            "document_id": document_id,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "completed"
        }