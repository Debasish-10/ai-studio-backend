
import time

import numpy as np
import torch

from sentence_transformers import CrossEncoder


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"

QUERY = (
    "How many annual leave days can "
    "employees carry forward?"
)

# Simulated retrieval order from a first-stage retriever.
CANDIDATES = [
    {
        "id": "chunk_1",
        "text": (
            "Employees receive 24 working days "
            "of annual leave each year."
        ),
    },
    {
        "id": "chunk_2",
        "text": (
            "Employees may carry forward up to "
            "10 unused annual leave days."
        ),
    },
    {
        "id": "chunk_3",
        "text": (
            "Employees receive 12 sick leave "
            "days every year."
        ),
    },
    {
        "id": "chunk_4",
        "text": (
            "Unused annual leave may be encashed "
            "during final settlement."
        ),
    },
    {
        "id": "chunk_5",
        "text": (
            "Remote work is permitted two "
            "days per week."
        ),
    },
]


def first_relevant_rank(items, relevant_id):
    for rank, item in enumerate(items, start=1):
        if item["id"] == relevant_id:
            return rank

    return None


def run_reranking(model, query, candidates, batch_size=8):
    pairs = [
        (query, item["text"])
        for item in candidates
    ]

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity()
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    ranked = []

    for candidate, score in zip(candidates, scores):
        ranked.append({
            **candidate,
            "rerank_score": float(score)
        })

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked, elapsed_ms


def main():
    model = CrossEncoder(MODEL_ID)

    print("MODEL:", MODEL_ID)
    print("MAX LENGTH:", model.max_length)

    print("\nBEFORE RERANKING")

    for rank, item in enumerate(CANDIDATES, 1):
        print(rank, item["id"], item["text"])

    # Warm-up; exclude initialization from measured inference time.
    model.predict(
        [(QUERY, CANDIDATES[0]["text"])],
        show_progress_bar=False
    )

    ranked, latency_ms = run_reranking(
        model,
        QUERY,
        CANDIDATES,
        batch_size=8
    )

    print("\nAFTER RERANKING")

    for rank, item in enumerate(ranked, 1):
        print(
            f"{rank}. {item['id']} "
            f"score={item['rerank_score']:.4f}"
        )
        print("   ", item["text"])

    relevant_id = "chunk_2"

    before_rank = first_relevant_rank(
        CANDIDATES,
        relevant_id
    )

    after_rank = first_relevant_rank(
        ranked,
        relevant_id
    )

    print("\nRANKING COMPARISON")
    print("Before rank:", before_rank)
    print("After rank:", after_rank)
    print("Reranking latency:", round(latency_ms, 2), "ms")

    print("\nBATCH SIZE COMPARISON")

    for batch_size in [1, 2, 4, 8]:
        _, elapsed = run_reranking(
            model,
            QUERY,
            CANDIDATES,
            batch_size=batch_size
        )

        print(
            f"Batch size {batch_size}: "
            f"{elapsed:.2f} ms"
        )


if __name__ == "__main__":
    main()
