import asyncio
import json
import unittest

from collections.abc import Iterator
from fastapi.testclient import TestClient

from app.main import app
from app.llm.provider import LLMProvider
from app.schemas.chat import ConversationMessage, StructuredChatResponse
from app.services.chat_service import ChatService, get_chat_service


class FakeLLMProvider(LLMProvider):
    def generate_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        return f"Experiment at {temperature}"

    def stream_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        return iter(())

    def generate_structured_output(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        return StructuredChatResponse(answer="Structured response")


class PromptsApiTests(unittest.TestCase):
    def test_build_prompt_returns_constructed_prompt(self) -> None:
        status_code, response_body = _post_json(
            "/prompts/build",
            {
                "system_instructions": "You are a helpful assistant.",
                "context": "Use concise language.",
                "examples": [{"input": "Hello", "output": "Hi there!"}],
                "user_instructions": "Greet the user.",
            },
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(
            response_body,
            {
                "prompt": "System Instructions:\nYou are a helpful assistant.\n\n"
                "Context:\nUse concise language.\n\n"
                "Examples:\nInput:\nHello\nOutput:\nHi there!\n\n"
                "User Instructions:\nGreet the user."
            },
        )

    def test_prompt_experiment_runs_a_one_shot_prompt_with_temperature(self) -> None:
        app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeLLMProvider())

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/prompts/experiment",
                    json={
                        "shot_type": "one_shot",
                        "system_instructions": "Follow the example.",
                        "context": "",
                        "examples": [{"input": "one", "output": "first"}],
                        "user_instructions": "Continue the pattern.",
                        "temperature": 0.35,
                    },
                )
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Experiment at 0.35"})

    def test_prompt_experiment_rejects_an_invalid_shot_example_count(self) -> None:
        status_code, _ = _post_json(
            "/prompts/experiment",
            {
                "shot_type": "one_shot",
                "system_instructions": "Follow the example.",
                "context": "",
                "examples": [],
                "user_instructions": "Continue the pattern.",
            },
        )

        self.assertEqual(status_code, 422)

    def test_prompt_comparison_returns_each_configuration_output(self) -> None:
        app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeLLMProvider())

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/prompts/compare",
                    json={
                        "user_instructions": "Continue the pattern.",
                        "configurations": [
                            {
                                "label": "Zero-shot",
                                "shot_type": "zero_shot",
                                "system_instructions": "Be concise.",
                                "context": "",
                                "temperature": 0.2,
                            },
                            {
                                "label": "One-shot",
                                "shot_type": "one_shot",
                                "system_instructions": "Follow the example.",
                                "context": "",
                                "examples": [{"input": "one", "output": "first"}],
                                "temperature": 0.8,
                            },
                        ],
                    },
                )
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "results": [
                    {
                        "label": "Zero-shot",
                        "prompt": "System Instructions:\nBe concise.\n\nContext:\n\n\n"
                        "Examples:\n\n\nUser Instructions:\nContinue the pattern.",
                        "output": "Experiment at 0.2",
                    },
                    {
                        "label": "One-shot",
                        "prompt": "System Instructions:\nFollow the example.\n\nContext:\n\n\n"
                        "Examples:\nInput:\none\nOutput:\nfirst\n\n"
                        "User Instructions:\nContinue the pattern.",
                        "output": "Experiment at 0.8",
                    },
                ]
            },
        )


def _post_json(path: str, payload: dict[str, object]) -> tuple[int, dict[str, str]]:
    return asyncio.run(_send_post_request(path, json.dumps(payload).encode()))


async def _send_post_request(
    path: str, body: bytes
) -> tuple[int, dict[str, str]]:
    sent_messages: list[dict[str, object]] = []
    request_received = False

    async def receive() -> dict[str, object]:
        nonlocal request_received

        if request_received:
            return {"type": "http.disconnect"}

        request_received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent_messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )

    status_code = next(
        message["status"]
        for message in sent_messages
        if message["type"] == "http.response.start"
    )
    response_content = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )

    return int(status_code), json.loads(response_content)
