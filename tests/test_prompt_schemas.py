import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.prompt import (
    Example,
    PromptCreate,
    PromptComparisonRequest,
    PromptExperimentRequest,
    PromptListQuery,
    PromptPage,
    PromptRead,
    PromptRequest,
    PromptResponse,
    PromptShotType,
    PromptUpdate,
    PromptVersionRead,
)


class PromptSchemaTests(unittest.TestCase):
    def test_prompt_request_accepts_examples(self) -> None:
        request = PromptRequest(
            system_instructions="You are a helpful assistant.",
            user_instructions="Summarize the provided text.",
            context="FastAPI uses Pydantic for data validation.",
            examples=[
                {
                    "input": "FastAPI is fast.",
                    "output": "FastAPI has high performance.",
                }
            ],
        )

        self.assertEqual(
            request.examples,
            [
                Example(
                    input="FastAPI is fast.",
                    output="FastAPI has high performance.",
                )
            ],
        )

    def test_prompt_request_requires_all_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PromptRequest(
                system_instructions="You are a helpful assistant.",
                user_instructions="Summarize the provided text.",
                context="",
            )

    def test_prompt_response_serializes_prompt(self) -> None:
        response = PromptResponse(prompt="System: You are a helpful assistant.")

        self.assertEqual(
            response.model_dump(),
            {"prompt": "System: You are a helpful assistant."},
        )

    def test_zero_shot_experiment_requires_no_examples(self) -> None:
        request = PromptExperimentRequest(
            shot_type=PromptShotType.ZERO_SHOT,
            system_instructions="You are helpful.",
            user_instructions="Greet the user.",
            context="",
        )

        self.assertEqual(request.examples, [])
        self.assertEqual(request.temperature, 1.0)

    def test_one_shot_experiment_requires_exactly_one_example(self) -> None:
        request = PromptExperimentRequest(
            shot_type=PromptShotType.ONE_SHOT,
            system_instructions="You are helpful.",
            user_instructions="Continue the pattern.",
            context="",
            examples=[{"input": "Hello", "output": "Hi"}],
            temperature=0.35,
        )

        self.assertEqual(len(request.examples), 1)
        self.assertEqual(request.temperature, 0.35)

    def test_few_shot_experiment_requires_at_least_two_examples(self) -> None:
        with self.assertRaises(ValidationError):
            PromptExperimentRequest(
                shot_type=PromptShotType.FEW_SHOT,
                system_instructions="You are helpful.",
                user_instructions="Continue the pattern.",
                context="",
                examples=[{"input": "one", "output": "first"}],
            )

    def test_zero_shot_experiment_rejects_examples(self) -> None:
        with self.assertRaises(ValidationError):
            PromptExperimentRequest(
                shot_type=PromptShotType.ZERO_SHOT,
                system_instructions="You are helpful.",
                user_instructions="Greet the user.",
                context="",
                examples=[{"input": "Hello", "output": "Hi"}],
            )

    def test_prompt_comparison_requires_at_least_two_configurations(self) -> None:
        with self.assertRaises(ValidationError):
            PromptComparisonRequest(
                user_instructions="Greet the user.",
                configurations=[
                    {
                        "label": "Zero-shot",
                        "shot_type": "zero_shot",
                        "system_instructions": "You are helpful.",
                        "context": "",
                    }
                ],
            )

    def test_prompt_persistence_schemas_validate_defaults_and_partial_updates(self) -> None:
        create = PromptCreate(title="Greeting", content="Say hello")
        update = PromptUpdate(description=None)

        self.assertEqual(create.category, "general")
        self.assertFalse(create.is_favorite)
        self.assertEqual(update.model_dump(exclude_unset=True), {"description": None})

    def test_prompt_persistence_schemas_reject_blank_and_empty_updates(self) -> None:
        with self.assertRaises(ValidationError):
            PromptCreate(title="   ", content="Say hello")
        with self.assertRaises(ValidationError):
            PromptUpdate()

    def test_prompt_read_accepts_orm_style_attributes(self) -> None:
        created_at = datetime.now(UTC)
        prompt = SimpleNamespace(
            id=uuid4(),
            title="Greeting",
            content="Say hello",
            description=None,
            category="general",
            is_favorite=False,
            is_template=False,
            created_at=created_at,
            updated_at=created_at,
        )

        response = PromptRead.model_validate(prompt)

        self.assertEqual(response.title, "Greeting")
        self.assertEqual(response.created_at, created_at)

    def test_prompt_version_read_contains_only_historical_content(self) -> None:
        created_at = datetime.now(UTC)
        version = SimpleNamespace(
            id=uuid4(),
            prompt_id=uuid4(),
            version=2,
            content="Say hello warmly",
            created_at=created_at,
        )

        response = PromptVersionRead.model_validate(version)

        self.assertEqual(response.content, "Say hello warmly")
        self.assertEqual(
            set(response.model_dump()),
            {"id", "prompt_id", "version", "content", "created_at"},
        )

    def test_prompt_list_query_normalizes_blank_search_and_validates_pagination(self) -> None:
        query = PromptListQuery(search="  ", category=" coding ", page=2, page_size=50)

        self.assertIsNone(query.search)
        self.assertEqual(query.category, "coding")
        self.assertEqual(query.page, 2)
        self.assertEqual(query.page_size, 50)
        with self.assertRaises(ValidationError):
            PromptListQuery(page=0)
        with self.assertRaises(ValidationError):
            PromptListQuery(page_size=101)
