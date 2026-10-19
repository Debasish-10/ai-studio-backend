
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


PDF_PATH = (
    Path(__file__).parent
    / "sample_docs"
    / "employee_handbook.pdf"
)

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
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk_number, text in enumerate(page_chunks, start=1):
            chunks.append({
                "text": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_number": chunk_number
                }
            })

    return chunks


def search_chunks(
    query,
    model,
    chunks,
    chunk_embeddings,
    top_k=3,
    threshold=0.50
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = np.dot(chunk_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        score = float(scores[index])

        if score < threshold:
            continue

        results.append({
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "score": score
        })

    return results


def main():
    # Document ingestion
    documents = load_pdf(PDF_PATH)
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No extractable text found in the PDF.")

    print("Total chunks:", len(chunks))

    # Generate document embeddings once
    model = SentenceTransformer(MODEL_NAME)

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    print("Embedding matrix:", chunk_embeddings.shape)

    # Interactive semantic search
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search_chunks(
            query=query,
            model=model,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            top_k=3,
            threshold=0.50
        )

        if not results:
            print("No sufficiently relevant information found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print("Similarity:", round(result["score"], 4))
            print("Metadata:", result["metadata"])
            print("Text:", result["text"])


if __name__ == "__main__":
    main()
