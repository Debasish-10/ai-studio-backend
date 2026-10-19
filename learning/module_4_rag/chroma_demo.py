
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Reuse your existing PDF processing functions
from faiss_demo import load_pdf, create_chunks


BASE_DIR = Path(__file__).parent

PDF_PATH = BASE_DIR / "sample_docs" / "employee_handbook.pdf"
DB_PATH = BASE_DIR / "chroma_data"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "employee_handbook_demo"


def create_collection():
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def ingest_pdf(collection, model):
    documents = load_pdf(PDF_PATH)
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No text chunks found in PDF.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"handbook_chunk_{index}")

        metadatas.append({
            **chunk["metadata"],
            "document_id": "handbook_001"
        })

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("Stored chunks:", collection.count())


def search_chroma(
    collection,
    model,
    query,
    top_k=3,
    similarity_threshold=0.30,
    document_id=None
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    filters = None

    if document_id is not None:
        filters = {"document_id": document_id}

    results = collection.query(
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
        # Our collection uses cosine distance
        similarity = 1.0 - float(distance)

        if similarity < similarity_threshold:
            continue

        matches.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
            "similarity": similarity
        })

    return matches


def main():
    model = SentenceTransformer(MODEL_NAME)
    collection = create_collection()

    # Only ingest if the collection is empty.
    # In production, use document versions instead.
    if collection.count() == 0:
        ingest_pdf(collection, model)
    else:
        print("Reusing persisted ChromaDB collection.")

    print("Total stored chunks:", collection.count())

    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        matches = search_chroma(
            collection=collection,
            model=model,
            query=query,
            top_k=3,
            similarity_threshold=0.20,
            document_id="handbook_001"
        )

        if not matches:
            print("No sufficiently relevant information found.")
            continue

        for rank, match in enumerate(matches, start=1):
            print(f"\n--- Result {rank} ---")
            print("ID:", match["id"])
            print(
                "Similarity:",
                round(match["similarity"], 4)
            )
            print("Metadata:", match["metadata"])
            print("Text:", match["text"])


if __name__ == "__main__":
    main()
