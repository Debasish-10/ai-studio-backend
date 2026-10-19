import pytest
from pydantic import ValidationError

from app.schemas.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunRequest,
)


def test_evaluation_dataset_accepts_cases_with_expected_outputs_and_criteria() -> None:
    dataset = EvaluationDataset(
        name="Greeting quality",
        cases=[
            {
                "input": "Greet Ada.",
                "expected_output": "Hello, Ada!",
                "criteria": ["Uses the name Ada", "Uses a friendly tone"],
            }
        ],
    )

    assert dataset.name == "Greeting quality"
    assert dataset.cases == [
        EvaluationCase(
            input="Greet Ada.",
            expected_output="Hello, Ada!",
            criteria=["Uses the name Ada", "Uses a friendly tone"],
        )
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input", ""),
        ("expected_output", ""),
        ("criteria", []),
    ],
)
def test_evaluation_case_rejects_missing_required_content(
    field: str,
    value: str | list[str],
) -> None:
    case = {
        "input": "Greet Ada.",
        "expected_output": "Hello, Ada!",
        "criteria": ["Uses the name Ada"],
    }
    case[field] = value

    with pytest.raises(ValidationError):
        EvaluationCase(**case)


def test_evaluation_dataset_requires_at_least_one_case() -> None:
    with pytest.raises(ValidationError):
        EvaluationDataset(name="Greeting quality", cases=[])


def test_evaluation_case_rejects_empty_criteria() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            input="Greet Ada.",
            expected_output="Hello, Ada!",
            criteria=[""],
        )


def test_evaluation_run_request_accepts_a_dataset_and_temperature() -> None:
    request = EvaluationRunRequest(
        dataset={
            "name": "Greeting quality",
            "cases": [
                {
                    "input": "Greet Ada.",
                    "expected_output": "Hello, Ada!",
                    "criteria": ["Uses the name Ada"],
                }
            ],
        },
        temperature=0.35,
    )

    assert request.dataset.name == "Greeting quality"
    assert request.temperature == 0.35
