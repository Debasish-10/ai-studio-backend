
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


# ----------------------------------
# 1. Sample documents
# ----------------------------------

documents = [
    Document(
        id="d1",
        page_content="Employees receive 24 annual leave days.",
        metadata={
            "owner_id": "user_a",
            "document_id": "handbook_a",
        },
    ),
    Document(
        id="d2",
        page_content="Employees can carry forward 10 leave days.",
        metadata={
            "owner_id": "user_a",
            "document_id": "handbook_a",
        },
    ),
    Document(
        id="d3",
        page_content="Confidential salary information.",
        metadata={
            "owner_id": "user_b",
            "document_id": "private_b",
        },
    ),
]


# ----------------------------------
# 2. Fake retriever for testing
# ----------------------------------

def retrieve(data: dict) -> list[Document]:
    question = data["question"].lower()
    trusted_user_id = data["trusted_user_id"]

    # Authorization happens BEFORE search.
    authorized = [
        doc for doc in documents
        if doc.metadata["owner_id"] == trusted_user_id
    ]

    if "carry forward" in question:
        relevant_id = "d2"
    elif "annual leave" in question:
        relevant_id = "d1"
    else:
        relevant_id = None

    if relevant_id is None:
        return []

    return [
        doc for doc in authorized
        if doc.id == relevant_id
    ]


retriever = RunnableLambda(retrieve)


# ----------------------------------
# 3. Retrieval evaluation metrics
# ----------------------------------

def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    return float(
        bool(set(retrieved_ids[:k]) & relevant_ids)
    )


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    for rank, doc_id in enumerate(
        retrieved_ids[:k], start=1
    ):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


# ----------------------------------
# 4. Golden evaluation dataset
# ----------------------------------

dataset = [
    {
        "question": "How many annual leave days?",
        "relevant_ids": {"d1"},
    },
    {
        "question": "How many days can I carry forward?",
        "relevant_ids": {"d2"},
    },
]


# ----------------------------------
# 5. Evaluate retrieval
# ----------------------------------

def run_evaluation():
    hits = []
    reciprocal_ranks = []

    for example in dataset:
        start = perf_counter()

        results = retriever.invoke(
            {
                "question": example["question"],
                "trusted_user_id": "user_a",
            },
            config={
                "run_name": "retrieval_eval",
                "tags": ["session11", "evaluation"],
            },
        )

        latency_ms = (
            perf_counter() - start
        ) * 1000

        ids = [doc.id for doc in results]

        hits.append(
            hit_at_k(
                ids,
                example["relevant_ids"],
                k=3,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank_at_k(
                ids,
                example["relevant_ids"],
                k=3,
            )
        )

        print("\nQuestion:", example["question"])
        print("Retrieved IDs:", ids)
        print("Latency:", round(latency_ms, 3), "ms")

    print("\nHit@3:", sum(hits) / len(hits))
    print(
        "MRR@3:",
        sum(reciprocal_ranks) / len(reciprocal_ranks),
    )


# ----------------------------------
# 6. Authorization isolation test
# ----------------------------------

def test_cross_user_isolation():
    results = retriever.invoke({
        "question": "annual leave",
        "trusted_user_id": "user_b",
    })

    assert all(
        doc.metadata["owner_id"] == "user_b"
        for doc in results
    )

    assert results == []

    print("\nCross-user isolation: PASSED")


if __name__ == "__main__":
    run_evaluation()
    test_cross_user_isolation()

