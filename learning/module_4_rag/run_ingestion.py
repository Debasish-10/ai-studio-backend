from rag_pipeline.config import PDF_PATH

from rag_pipeline.loaders.pdf_loader import PDFLoader
from rag_pipeline.chunking.text_chunker import TextChunker
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.ingestion import IngestionService


def main():
    vector_store = ChromaStore()

    ingestion_service = IngestionService(
        loader=PDFLoader(),
        chunker=TextChunker(),
        embedding_service=EmbeddingService(),
        vector_store=vector_store
    )

    result = ingestion_service.ingest(
        file_path=PDF_PATH,
        document_id="handbook_001"
    )

    print("Ingestion result:", result)
    print("Total stored chunks:", vector_store.count())


if __name__ == "__main__":
    main()