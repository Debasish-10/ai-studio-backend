import unittest

from app.schemas.chat import ChatResponse
from app.schemas.prompt import (
    Example,
    PromptComparisonRequest,
    PromptExperimentRequest,
    PromptRequest,
    PromptShotType,
)
from app.services.prompt_service import (
    build_prompt,
    compare_prompt_outputs,
    run_prompt_experiment,
)


class StubChatService:
    def __init__(self) -> None:
        self.received_message: str | None = None
        self.received_temperature: float | None = None

    def create_response(
        self,
        message: str,
        history: list[object] | None = None,
        temperature: float = 1.0,
    ) -> ChatResponse:
        self.received_message = message
        self.received_temperature = temperature
        return ChatResponse(message="Experiment response")


class PromptServiceTests(unittest.TestCase):
    def test_build_prompt_includes_all_populated_sections(self) -> None:
        response = build_prompt(
            PromptRequest(
                system_instructions="You are a helpful assistant.",
                context="Use concise language.",
                examples=[Example(input="Hello", output="Hi there!")],
                user_instructions="Greet the user.",
            )
        )

        self.assertEqual(
            response.prompt,
            "System Instructions:\nYou are a helpful assistant.\n\n"
            "Context:\nUse concise language.\n\n"
            "Examples:\nInput:\nHello\nOutput:\nHi there!\n\n"
            "User Instructions:\nGreet the user.",
        )

    def test_build_prompt_preserves_multiple_examples_in_order(self) -> None:
        response = build_prompt(
            PromptRequest(
                system_instructions="Follow examples.",
                context="",
                examples=[
                    Example(input="one", output="first"),
                    Example(input="two", output="second"),
                ],
                user_instructions="Continue the pattern.",
            )
        )

        self.assertIn("Input:\none\nOutput:\nfirst", response.prompt)
        self.assertIn("Input:\ntwo\nOutput:\nsecond", response.prompt)
        self.assertLess(response.prompt.index("Input:\none"), response.prompt.index("Input:\ntwo"))

    def test_build_prompt_keeps_empty_context_section(self) -> None:
        response = build_prompt(
            PromptRequest(
                system_instructions="System.",
                context="",
                examples=[],
                user_instructions="User.",
            )
        )

        self.assertIn("Context:\n\n\nExamples:", response.prompt)

    def test_build_prompt_keeps_empty_examples_section(self) -> None:
        response = build_prompt(
            PromptRequest(
                system_instructions="System.",
                context="Context.",
                examples=[],
                user_instructions="User.",
            )
        )

        self.assertIn("Examples:\n\n\nUser Instructions:", response.prompt)

    def test_run_prompt_experiment_uses_the_built_prompt_and_temperature(self) -> None:
        chat_service = StubChatService()
        response = run_prompt_experiment(
            PromptExperimentRequest(
                shot_type=PromptShotType.ONE_SHOT,
                system_instructions="Follow the example.",
                context="",
                examples=[Example(input="one", output="first")],
                user_instructions="Continue the pattern.",
                temperature=0.35,
            ),
            chat_service,
        )

        self.assertEqual(response.message, "Experiment response")
        self.assertEqual(chat_service.received_temperature, 0.35)
        self.assertIn("Input:\none\nOutput:\nfirst", chat_service.received_message)

    def test_compare_prompt_outputs_uses_the_same_input_for_each_configuration(self) -> None:
        chat_service = StubChatService()

        response = compare_prompt_outputs(
            PromptComparisonRequest(
                user_instructions="Continue the pattern.",
                configurations=[
                    {
                        "label": "Zero-shot",
                        "shot_type": "zero_shot",
                        "system_instructions": "Be concise.",
                        "context": "",
                        "temperature": 0.2,
                    },
                    {
                        "label": "Few-shot",
                        "shot_type": "few_shot",
                        "system_instructions": "Follow the examples.",
                        "context": "",
                        "examples": [
                            {"input": "one", "output": "first"},
                            {"input": "two", "output": "second"},
                        ],
                        "temperature": 0.8,
                    },
                ],
            ),
            chat_service,
        )

        self.assertEqual([result.label for result in response.results], ["Zero-shot", "Few-shot"])
        self.assertIn("User Instructions:\nContinue the pattern.", response.results[0].prompt)
        self.assertIn("Input:\none\nOutput:\nfirst", response.results[1].prompt)
        self.assertEqual(chat_service.received_temperature, 0.8)
