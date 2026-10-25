
from generation_eval_dataset import (
    GENERATION_EVAL_CASES
)

from rag_pipeline.evaluation.generation_metrics import GenerationEvaluator

# Extract this factory function from your
# existing run_rag_generation.py.
from run_rag_generation import create_rag_service


def main():
    rag = create_rag_service()
    evaluator = GenerationEvaluator()

    filters = {"document_id": "handbook_001"}

    results = []

    for case in GENERATION_EVAL_CASES:

        print("\n" + "=" * 60)
        print("QUESTION:", case["question"])

        response = rag.generate(
            query=case["question"],
            filters=filters,
            top_k=5
        )

        evaluation = evaluator.evaluate(
            case=case,
            answer=response.answer,
            citation_count=len(response.citations)
        )

        results.append(evaluation)

        print("ANSWER:", response.answer)
        print("STATUS:", response.status)
        print(
            "FACT COVERAGE:",
            evaluation.fact_coverage
        )
        print(
            "HAS RESOLVED CITATION:",
            evaluation.has_valid_citation
        )
        print(
            "ABSTENTION CORRECT:",
            evaluation.abstention_correct
        )

    abstention_accuracy = sum(
        result.abstention_correct
        for result in results
    ) / len(results)

    print("\n" + "=" * 60)
    print(
        "ABSTENTION CLASSIFICATION ACCURACY:",
        round(abstention_accuracy, 3)
    )


if __name__ == "__main__":
    main()
