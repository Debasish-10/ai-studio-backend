
import numpy as np

from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    query = "How many annual leave days?"

    passage = (
        "Employees receive 24 working days of "
        "paid annual leave per calendar year."
    )

    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True
    )

    document_embedding = model.encode_document(
        passage,
        normalize_embeddings=True
    )

    ordinary_query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    similarity = np.dot(
        query_embedding,
        document_embedding
    )

    print("Query shape:", query_embedding.shape)
    print("Document shape:", document_embedding.shape)
    print("Cosine similarity:", similarity)

    print(
        "Query method equals encode():",
        np.allclose(
            query_embedding,
            ordinary_query_embedding
        )
    )

    print("Available prompts:", model.prompts)


if __name__ == "__main__":
    main()
