
import numpy as np

from sentence_transformers import SentenceTransformer


MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5"
]

QUERY = (
    "Can employees carry unused vacation "
    "days forward?"
)

PASSAGES = [
    (
        "Employees may carry forward up to "
        "10 unused annual leave days."
    ),
    (
        "Employees can work remotely up to "
        "two days per week."
    ),
    (
        "The company provides medical "
        "insurance coverage."
    )
]


def compare_model(model_name: str):
    model = SentenceTransformer(model_name)

    query_embedding = model.encode_query(
        QUERY,
        normalize_embeddings=True
    )

    document_embeddings = model.encode_document(
        PASSAGES,
        normalize_embeddings=True
    )

    similarities = (
        document_embeddings @ query_embedding
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    print(f"\nMODEL: {model_name}")
    print(
        "Dimension:",
        model.get_sentence_embedding_dimension()
    )
    print("Max tokens:", model.max_seq_length)

    for rank, idx in enumerate(ranked_indices, 1):
        print(
            f"{rank}. Score: "
            f"{similarities[idx]:.4f}"
        )
        print("   ", PASSAGES[idx])


def main():
    for model_name in MODELS:
        compare_model(model_name)


if __name__ == "__main__":
    main()
