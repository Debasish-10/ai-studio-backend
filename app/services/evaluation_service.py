from app.schemas.evaluation import (
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationRunResponse,
)
from app.services.chat_service import ChatService


def run_evaluation(
    dataset: EvaluationDataset,
    chat_service: ChatService,
    temperature: float = 1.0,
) -> EvaluationRunResponse:
    """Run each evaluation case and compare actual output with expected output."""

    results = []
    for case in dataset.cases:
        response = chat_service.create_response(case.input, temperature=temperature)
        results.append(
            EvaluationCaseResult(
                input=case.input,
                expected_output=case.expected_output,
                actual_output=response.message,
                criteria=case.criteria,
                passed=(response.message.strip().lower()== case.expected_output.strip().lower())
            )
        )

    return EvaluationRunResponse(dataset_name=dataset.name, results=results)
