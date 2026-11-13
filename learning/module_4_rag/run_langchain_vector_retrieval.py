
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR / "sample_docs" / "employee_handbook.pdf"
)

CHROMA_PATH = BASE_DIR / "chroma_langchain_session11"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "handbook_session11"


# 1. Load and chunk documents.
def prepare_documents():
    loader = PyMuPDFLoader(
        str(PDF_PATH),
        mode="page",
    )

    pages = loader.load()

    for page in pages:
        page.metadata["document_id"] = "handbook_001"
        page.metadata["document_version"] = 1

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    splitter = (
        RecursiveCharacterTextSplitter
        .from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=120,
            chunk_overlap=24,
            add_start_index=True,
        )
    )

    chunks = splitter.split_documents(pages)

    # Deterministic IDs for this unchanged
    # learning PDF and chunking configuration.
    for index, chunk in enumerate(chunks):
        chunk.id = f"handbook_v1_chunk_{index:04d}"

    return chunks


# 2. Configure the embedding model.
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


# 3. Connect to ChromaDB.
def create_vector_store(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_configuration={
            "hnsw": {"space": "cosine"}
        },
    )


# 4. Display retrieved documents.
def display_results(title, results):
    print(f"\n--- {title} ---")

    for index, doc in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("ID:", doc.id)
        print("Page:", doc.metadata.get("page"))
        print("Text:", doc.page_content[:350])


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # Ingestion
    chunks = prepare_documents()

    print("Total chunks:", len(chunks))

    embeddings = create_embeddings()

    vector_store = create_vector_store(embeddings)

    ids = [chunk.id for chunk in chunks]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print("Stored chunks:", len(ids))

    # Query
    query = (
        "How many unused annual leave days "
        "can employees carry forward?"
    )

    # A. Direct vector store search
    direct_results = vector_store.similarity_search(
        query,
        k=3,
    )

    display_results(
        "DIRECT SIMILARITY SEARCH",
        direct_results,
    )

    # B. LangChain retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    retriever_results = retriever.invoke(query)

    display_results(
        "RETRIEVER RESULTS",
        retriever_results,
    )

    # C. Similarity search with cosine distances
    scored_results = (
        vector_store.similarity_search_with_score(
            query,
            k=3,
        )
    )

    print("\n--- COSINE DISTANCES ---")
    for doc, distance in scored_results:
        print(
            "Distance:",
            round(distance, 4),
            "Page:",
            doc.metadata.get("page"),
        )

    # D. Metadata filtering
    filtered_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {
                "document_id": "handbook_001"
            },
        },
    )

    filtered_results = filtered_retriever.invoke(query)

    display_results(
        "METADATA FILTERED RESULTS",
        filtered_results,
    )

    # E. MMR retriever
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    mmr_results = mmr_retriever.invoke(query)

    display_results(
        "MMR RESULTS",
        mmr_results,
    )


if __name__ == "__main__":
    main()
