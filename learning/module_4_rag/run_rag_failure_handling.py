
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    ANSWERED = "answered"
    NO_EVIDENCE = "no_evidence"
    CONFLICT = "conflict"
    FORBIDDEN = "forbidden"
    UNAVAILABLE = "unavailable"


@dataclass
class Evidence:
    source_id: str
    text: str
    similarity: float
    authorized: bool = True
    active: bool = True


# Illustrative threshold only; calibrate using evals.
MIN_SIMILARITY = 0.70


def retrieve(question: str) -> list[Evidence]:
    q = question.lower()

    if "service failure" in q:
        raise ConnectionError("Vector store unavailable")

    if "private" in q:
        return [
            Evidence(
                "PRIVATE-1",
                "Confidential employee compensation.",
                0.95,
                authorized=False,
            )
        ]

    if "stock option" in q:
        return [
            Evidence(
                "DOC-3",
                "Employees receive annual leave.",
                0.28,
            )
        ]

    if "conflicting" in q:
        return [
            Evidence(
                "POLICY-A",
                "Carry-forward limit: 10 days.",
                0.91,
            ),
            Evidence(
                "POLICY-B",
                "Carry-forward limit: 15 days.",
                0.89,
            ),
        ]

    return [
        Evidence(
            "POLICY-2025",
            "Carry-forward limit: 10 days.",
            0.93,
        )
    ]


def rerank(evidence: list[Evidence],
           simulate_failure: bool = False):
    if simulate_failure:
        raise TimeoutError("Reranker timed out")

    return sorted(
        evidence,
        key=lambda e: e.similarity,
        reverse=True,
    )


def response(status, answer, sources=None):
    return {
        "status": status.value,
        "answer": answer,
        "sources": sources or [],
    }


def run_rag(question: str,
            reranker_failure: bool = False):

    # Demonstration only. Real authorization must
    # be enforced before querying the vector store.
    try:
        evidence = retrieve(question)
    except ConnectionError:
        return response(
            Status.UNAVAILABLE,
            "Retrieval service is temporarily unavailable.",
        )

    # Defense-in-depth check only:
    # unauthorized evidence must never reach the LLM.
    if any(not item.authorized for item in evidence):
        return response(
            Status.FORBIDDEN,
            "Document access denied.",
        )

    evidence = [
        item for item in evidence
        if item.active
    ]

    evidence = [
        item for item in evidence
        if item.similarity >= MIN_SIMILARITY
    ]

    if not evidence:
        return response(
            Status.NO_EVIDENCE,
            "Insufficient relevant evidence.",
        )

    try:
        evidence = rerank(
            evidence,
            simulate_failure=reranker_failure,
        )
    except TimeoutError:
        # Safe fallback: original authorized evidence.
        pass

    # Controlled conflict fixture for this experiment.
    if (
        "conflicting" in question.lower()
        and len(evidence) >= 2
    ):
        return response(
            Status.CONFLICT,
            "Sources disagree: 10 days vs 15 days.",
            [e.source_id for e in evidence],
        )

    best = evidence[0]

    return response(
        Status.ANSWERED,
        best.text,
        [best.source_id],
    )


def main():
    scenarios = [
        ("Carry-forward policy", False),
        ("Stock option vesting", False),
        ("Conflicting leave policy", False),
        ("Carry-forward policy", True),
        ("Private compensation details", False),
        ("Service failure", False),
    ]

    for question, fail_reranker in scenarios:
        result = run_rag(
            question,
            reranker_failure=fail_reranker,
        )

        print("\nQUESTION:", question)
        print("RESULT:", result)


if __name__ == "__main__":
    main()
