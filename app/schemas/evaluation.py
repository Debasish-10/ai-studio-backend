from typing import Annotated

from pydantic import BaseModel, Field

EvaluationCriterion = Annotated[str, Field(min_length=1, max_length=500)]


class EvaluationCase(BaseModel):
    """One input, expected output, and criteria for a future evaluation run."""

    input: str = Field(min_length=1, max_length=4_000)
    expected_output: str = Field(min_length=1, max_length=4_000)
    criteria: list[EvaluationCriterion] = Field(min_length=1)


class EvaluationDataset(BaseModel):
    """A named collection of evaluation cases."""

    name: str = Field(min_length=1, max_length=100)
    cases: list[EvaluationCase] = Field(min_length=1)


class EvaluationRunRequest(BaseModel):
    dataset: EvaluationDataset
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)


class EvaluationCaseResult(BaseModel):
    input: str
    expected_output: str
    actual_output: str
    criteria: list[EvaluationCriterion]
    passed: bool


class EvaluationRunResponse(BaseModel):
    dataset_name: str
    results: list[EvaluationCaseResult]
