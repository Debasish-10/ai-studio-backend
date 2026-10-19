
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent

PDF_PATH = BASE_DIR / "sample_docs" / "employee_handbook.pdf"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_pdf(path: Path):
    reader = PdfReader(path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": path.name,
                "page": page_number
            }
        })

    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk_number, text in enumerate(
            page_chunks,
            start=1
        ):
            chunks.append({
                "text": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_number": chunk_number
                }
            })

    return chunks


def create_faiss_index(chunks, model):
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings


def search_faiss(
    query,
    model,
    index,
    chunks,
    top_k=3,
    threshold=0.50
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(score)
        })

    return results


def main():
    # 1. Read and chunk the PDF
    documents = load_pdf(PDF_PATH)
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No extractable PDF chunks found.")

    print("Total chunks:", len(chunks))

    # 2. Load the embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 3. Build the FAISS index
    index, embeddings = create_faiss_index(chunks, model)

    print("Embedding shape:", embeddings.shape)
    print("FAISS index size:", index.ntotal)

    # 4. Search
    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search_faiss(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=3,
            threshold=0.50
        )

        if not results:
            print("No sufficiently relevant information found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print("Score:", round(result["score"], 4))
            print("Metadata:", result["metadata"])
            print("Text:", result["text"])


if __name__ == "__main__":
    main()
