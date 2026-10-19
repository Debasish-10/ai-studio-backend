
def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved_ids[:k]

    return float(
        any(chunk_id in relevant_ids for chunk_id in top_k)
    )


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError(
            "Recall requires at least one relevant chunk"
        )

    retrieved = set(retrieved_ids[:k])

    return len(
        retrieved.intersection(relevant_ids)
    ) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5
) -> dict:
    return {
        "hit": hit_at_k(retrieved_ids, relevant_ids, k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": reciprocal_rank_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )
    }
