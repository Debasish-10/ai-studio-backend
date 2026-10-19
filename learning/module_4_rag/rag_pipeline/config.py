from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "sample_docs" / "employee_handbook.pdf"

CHROMA_PATH = BASE_DIR / "rag_chroma_data"

COLLECTION_NAME = "rag_pipeline_chunks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100