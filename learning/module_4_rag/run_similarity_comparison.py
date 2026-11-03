
import numpy as np

from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

QUERY = "How many annual leave days do employees receive?"

DOCUMENTS = [
    "Employees receive 24 working days of annual leave.",
    "Employees can work remotely two days per week.",
    "Employees receive medical insurance benefits.",
]


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def main():
    model = SentenceTransformer(MODEL_ID)

    # Raw, non-normalized embeddings
    query = model.encode_query(
        QUERY,
        normalize_embeddings=False
    )

    documents = model.encode_document(
        DOCUMENTS,
        normalize_embeddings=False
    )

    print("\nRAW EMBEDDING COMPARISON")

    for i, document in enumerate(DOCUMENTS):
        vector = documents[i]

        cosine = cosine_similarity(query, vector)
        dot = np.dot(query, vector)
        euclidean = np.linalg.norm(query - vector)

        print(f"\nDocument {i + 1}: {document}")
        print(f"Cosine:    {cosine:.4f}")
        print(f"Dot:       {dot:.4f}")
        print(f"Euclidean: {euclidean:.4f}")

    # Normalize both query and documents
    query_norm = query / np.linalg.norm(query)

    documents_norm = documents / np.linalg.norm(
        documents, axis=1, keepdims=True
    )

    print("\nNORMALIZED EMBEDDING COMPARISON")

    cosine_scores = np.array([
        cosine_similarity(query_norm, vector)
        for vector in documents_norm
    ])

    dot_scores = documents_norm @ query_norm

    distances = np.linalg.norm(
        documents_norm - query_norm,
        axis=1
    )

    print("Cosine scores:", np.round(cosine_scores, 4))
    print("Dot scores:", np.round(dot_scores, 4))
    print("Distances:", np.round(distances, 4))

    print(
        "Cosine and dot equal:",
        np.allclose(cosine_scores, dot_scores)
    )

    print("\nRANKINGS")

    print(
        "Cosine (high to low):",
        np.argsort(-cosine_scores) + 1
    )

    print(
        "Dot (high to low):",
        np.argsort(-dot_scores) + 1
    )

    print(
        "Euclidean (low to high):",
        np.argsort(distances) + 1
    )


if __name__ == "__main__":
    main()
