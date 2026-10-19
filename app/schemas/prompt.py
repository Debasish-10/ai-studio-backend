from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PromptCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    description: str | None = None
    category: str = Field(default="general", min_length=1, max_length=100)
    is_favorite: bool = False
    is_template: bool = False

    @field_validator("title", "content", "category")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank.")
        return value


class PromptUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    content: str | None = Field(default=None, min_length=1)
    description: str | None = None
    category: str | None = Field(default=None, min_length=1, max_length=100)
    is_favorite: bool | None = None
    is_template: bool | None = None

    @field_validator("title", "content", "category")
    @classmethod
    def validate_non_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank.")
        return value

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "PromptUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided.")
        return self


class PromptRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    content: str
    description: str | None
    category: str
    is_favorite: bool
    is_template: bool
    created_at: datetime
    updated_at: datetime


class PromptListQuery(BaseModel):
    search: str | None = Field(default=None, max_length=500)
    category: str | None = Field(default=None, max_length=100)
    is_favorite: bool | None = None
    is_template: bool | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @field_validator("search", "category", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            return value.strip() or None
        return value


class PromptPage(BaseModel):
    items: list[PromptRead]
    page: int
    page_size: int
    total: int
    total_pages: int


class PromptVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    prompt_id: UUID
    version: int
    content: str
    created_at: datetime


class Example(BaseModel):
    input: str
    output: str


class PromptRequest(BaseModel):
    system_instructions: str
    user_instructions: str
    context: str
    examples: list[Example]


class PromptResponse(BaseModel):
    prompt: str


class PromptShotType(str, Enum):
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"


class PromptExperimentConfig(BaseModel):
    shot_type: PromptShotType
    system_instructions: str
    context: str
    examples: list[Example] = Field(default_factory=list)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_examples_for_shot_type(self) -> "PromptExperimentConfig":
        example_count = len(self.examples)

        if self.shot_type == PromptShotType.ZERO_SHOT and example_count != 0:
            raise ValueError("zero_shot experiments cannot include examples.")
        if self.shot_type == PromptShotType.ONE_SHOT and example_count != 1:
            raise ValueError("one_shot experiments require exactly one example.")
        if self.shot_type == PromptShotType.FEW_SHOT and example_count < 2:
            raise ValueError("few_shot experiments require at least two examples.")

        return self


class PromptExperimentRequest(PromptExperimentConfig):
    user_instructions: str


class PromptComparisonConfiguration(PromptExperimentConfig):
    label: str = Field(min_length=1, max_length=100)


class PromptComparisonRequest(BaseModel):
    user_instructions: str
    configurations: list[PromptComparisonConfiguration] = Field(min_length=2)


class PromptComparisonResult(BaseModel):
    label: str
    prompt: str
    output: str


class PromptComparisonResponse(BaseModel):
    results: list[PromptComparisonResult]
