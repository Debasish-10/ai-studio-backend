
from dataclasses import dataclass


K = 3


@dataclass
class EvalCase:
    question: str
    relevant_ids: set[str]
    baseline: list[str]
    improved: list[str]
    should_abstain: bool = False


# -------------------------------------
# 1. Evaluation dataset
# -------------------------------------

eval_cases = [
    EvalCase(
        question="What is the leave carry-forward policy?",
        relevant_ids={"C2", "C5"},
        baseline=["C3", "C2", "C8"],
        improved=["C2", "C5", "C3"],
    ),
    EvalCase(
        question="How many annual leave days are allowed?",
        relevant_ids={"C7"},
        baseline=["C8", "C7", "C4"],
        improved=["C7", "C8", "C4"],
    ),
    EvalCase(
        question="What deployment policy applies to Atlas?",
        relevant_ids={"C10", "C11"},
        baseline=["C10", "C4", "C7"],
        improved=["C10", "C11", "C4"],
    ),
    EvalCase(
        question="What is the stock-option vesting period?",
        relevant_ids=set(),
        baseline=["C1", "C3", "C4"],
        improved=["C3", "C4", "C1"],
        should_abstain=True,
    ),
]


# -------------------------------------
# 2. Retrieval metrics
# -------------------------------------

def recall_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def precision_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    found = set(retrieved[:k]) & relevant
    return len(found) / k


def reciprocal_rank_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    for rank, chunk_id in enumerate(
        retrieved[:k], start=1
    ):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def complete_coverage(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> bool:
    return relevant.issubset(set(retrieved[:k]))


# -------------------------------------
# 3. Evaluate retrieval strategy
# -------------------------------------

def evaluate(strategy: str):
    rows = []

    for case in eval_cases:
        retrieved = getattr(case, strategy)

        # No-answer queries need separate evaluation.
        if case.should_abstain:
            continue

        recall = recall_at_k(
            case.relevant_ids, retrieved, K
        )
        precision = precision_at_k(
            case.relevant_ids, retrieved, K
        )
        mrr = reciprocal_rank_at_k(
            case.relevant_ids, retrieved, K
        )
        coverage = complete_coverage(
            case.relevant_ids, retrieved, K
        )

        rows.append({
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "coverage": int(coverage),
        })

        print(f"\nQuestion: {case.question}")
        print("Retrieved:", retrieved[:K])
        print("Recall@3:", round(recall, 3))
        print("Precision@3:", round(precision, 3))
        print("RR@3:", round(mrr, 3))
        print("Complete coverage:", coverage)

    print("\n--- AGGREGATE METRICS ---")

    for metric in [
        "recall",
        "precision",
        "mrr",
        "coverage",
    ]:
        average = sum(
            row[metric] for row in rows
        ) / len(rows)

        print(
            f"{metric}: {average:.3f}"
        )


# -------------------------------------
# 4. Execute comparison
# -------------------------------------

def main():
    print("\n========== BASELINE ==========")
    evaluate("baseline")

    print("\n========== IMPROVED ==========")
    evaluate("improved")

    print("\nNO-ANSWER TEST CASES:")
    for case in eval_cases:
        if case.should_abstain:
            print(case.question)
            print(
                "Expected behavior: abstain "
                "because no supporting evidence exists."
            )
            print(
                "Important: this still needs a "
                "real evidence sufficiency gate."
            )


if __name__ == "__main__":
    main()
