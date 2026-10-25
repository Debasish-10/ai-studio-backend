
from dataclasses import dataclass


@dataclass
class GenerationEvaluationResult:
    question_id: str
    fact_coverage: float | None
    has_valid_citation: bool
    abstained: bool
    abstention_correct: bool


class GenerationEvaluator:

    ABSTENTION_PHRASES = (
        "does not specify",
        "don't have sufficient information",
        "couldn't find sufficient information",
        "not enough information",
        "does not contain enough information",
        "cannot determine"
    )

    def evaluate(
        self,
        case: dict,
        answer: str,
        citation_count: int
    ) -> GenerationEvaluationResult:

        answer_lower = answer.lower()

        required_facts = case["required_facts"]

        if required_facts:
            found = sum(
                fact.lower() in answer_lower
                for fact in required_facts
            )

            coverage = found / len(required_facts)
        else:
            coverage = None

        abstained = any(
            phrase in answer_lower
            for phrase in self.ABSTENTION_PHRASES
        )

        return GenerationEvaluationResult(
            question_id=case["id"],
            fact_coverage=coverage,
            has_valid_citation=(citation_count > 0),
            abstained=abstained,
            abstention_correct=(
                abstained == case["should_abstain"]
            )
        )
