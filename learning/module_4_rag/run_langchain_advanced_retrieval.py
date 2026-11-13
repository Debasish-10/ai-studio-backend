
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.retrievers import (
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)

from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR / "sample_docs" / "employee_handbook.pdf"
)

CHROMA_PATH = (
    BASE_DIR / "chroma_langchain_session11"
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "handbook_session11"


# ----------------------------------------
# 1. Load the same handbook and chunk it
# ----------------------------------------

def load_chunks():
    pages = PyMuPDFLoader(
        str(PDF_PATH),
        mode="page",
    ).load()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"handbook_v1_chunk_{index:04d}"
        )
        chunk.metadata["document_id"] = "handbook_001"

    return chunks


# ----------------------------------------
# 2. Connect to existing ChromaDB
# ----------------------------------------

def create_semantic_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    if vector_store._collection.count() == 0:
        raise RuntimeError(
            "Chroma collection is empty. "
            "Run Part 3 ingestion first."
        )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8,
            "filter": {
                "document_id": "handbook_001",
            },
        },
    )


# ----------------------------------------
# 3. Configure BM25
# ----------------------------------------

def create_bm25_retriever(chunks):
    retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=lambda text: text.lower().split(),
    )

    retriever.k = 8

    return retriever


# ----------------------------------------
# 4. Hybrid retrieval with RRF
# ----------------------------------------

def create_hybrid_retriever(bm25, semantic):
    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.5, 0.5],
        # Use content identity for this demo because
        # the existing Part 3 Chroma index did not
        # store chunk_id as metadata.
    )


# ----------------------------------------
# 5. Cross-Encoder reranker
# ----------------------------------------

def create_reranking_retriever(hybrid):
    model = HuggingFaceCrossEncoder(
        model_name=(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
    )

    reranker = CrossEncoderReranker(
        model=model,
        top_n=5,
    )

    return ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=reranker,
    )


# ----------------------------------------
# 6. Print retrieved passages
# ----------------------------------------

def show_results(title, docs):
    print(f"\n=== {title} ===")

    for rank, doc in enumerate(docs, start=1):
        print(f"\nRank {rank}")
        print("Page:", doc.metadata.get("page"))
        print("Text:", doc.page_content[:220])


# ----------------------------------------
# 7. Run the experiment
# ----------------------------------------

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    if not CHROMA_PATH.exists():
        raise RuntimeError(
            "Run the Part 3 Chroma ingestion first."
        )

    chunks = load_chunks()

    print("BM25 indexed chunks:", len(chunks))

    bm25 = create_bm25_retriever(chunks)
    semantic = create_semantic_retriever()

    hybrid = create_hybrid_retriever(
        bm25,
        semantic,
    )

    reranked = create_reranking_retriever(
        hybrid
    )

    query = (
        "How many unused annual leave days "
        "can employees carry forward?"
    )

    show_results(
        "BM25",
        bm25.invoke(query),
    )

    show_results(
        "SEMANTIC",
        semantic.invoke(query),
    )

    show_results(
        "HYBRID RRF",
        hybrid.invoke(query)[:5],
    )

    show_results(
        "HYBRID + CROSS-ENCODER",
        reranked.invoke(query),
    )


if __name__ == "__main__":
    main()
