
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_ID, device="cpu")


# A small fictional employee handbook dataset.
# The parent summaries are known from the document.
chunks = [
    {
        "id": "C1",
        "parent_id": "P1",
        "title": "Annual Leave Policy",
        "context": (
            "Employee handbook, Annual Leave Policy. "
            "This passage describes the expiry date "
            "for carried-forward unused annual leave."
        ),
        "text": (
            "This balance must be utilized before "
            "March 31 of the following year."
        ),
    },
    {
        "id": "C2",
        "parent_id": "P2",
        "title": "Expense Reimbursement Policy",
        "context": (
            "Employee handbook, Expense Reimbursement. "
            "This passage explains the deadline for "
            "submitting travel reimbursement claims."
        ),
        "text": (
            "These requests must be submitted "
            "within 30 days."
        ),
    },
    {
        "id": "C3",
        "parent_id": "P3",
        "title": "Remote Work Policy",
        "context": (
            "Employee handbook, Remote Work Policy. "
            "This passage describes manager approval "
            "for remote working arrangements."
        ),
        "text": (
            "The arrangement requires approval "
            "from the reporting manager."
        ),
    },
    {
        "id": "C4",
        "parent_id": "P4",
        "title": "Annual Leave Entitlement",
        "context": (
            "Employee handbook, Annual Leave Entitlement. "
            "This passage describes the annual allocation "
            "of paid vacation days."
        ),
        "text": (
            "Employees receive 24 working days "
            "each calendar year."
        ),
    },
]


def embed(texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def rank_documents(query, document_vectors):
    query_vector = embed([query])[0]

    # Both document and query vectors are normalized.
    scores = document_vectors @ query_vector

    return np.argsort(scores)[::-1], scores


def display_results(title, order, scores):
    print(f"\n=== {title} ===")

    for rank, index in enumerate(order, start=1):
        chunk = chunks[index]

        print(
            f"{rank}. {chunk['id']} | "
            f"{chunk['title']} | "
            f"score={scores[index]:.4f}"
        )


def main():
    query = (
        "When does unused annual leave "
        "carried forward from last year expire?"
    )

    # A. Original text only
    original_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # B. Context + original text
    contextual_texts = [
        f"{chunk['context']}\n\n{chunk['text']}"
        for chunk in chunks
    ]

    original_vectors = embed(original_texts)
    contextual_vectors = embed(contextual_texts)

    original_order, original_scores = rank_documents(
        query,
        original_vectors,
    )

    contextual_order, contextual_scores = rank_documents(
        query,
        contextual_vectors,
    )

    print("QUESTION:", query)

    display_results(
        "NORMAL EMBEDDINGS",
        original_order,
        original_scores,
    )

    display_results(
        "CONTEXTUAL EMBEDDINGS",
        contextual_order,
        contextual_scores,
    )

    # Ground truth for this small synthetic example.
    relevant_id = "C1"

    normal_ids = [
        chunks[i]["id"] for i in original_order
    ]

    contextual_ids = [
        chunks[i]["id"] for i in contextual_order
    ]

    print("\nRelevant chunk:", relevant_id)

    print(
        "Normal rank:",
        normal_ids.index(relevant_id) + 1,
    )

    print(
        "Contextual rank:",
        contextual_ids.index(relevant_id) + 1,
    )


if __name__ == "__main__":
    main()
