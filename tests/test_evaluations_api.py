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
        return f"Actual output at {temperature}: {messages[-1].content}"

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


def test_evaluation_run_returns_structured_case_results() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeLLMProvider())

    try:
        with TestClient(app) as client:
            response = client.post(
                "/evaluations/run",
                json={
                    "dataset": {
                        "name": "Greeting quality",
                        "cases": [
                            {
                                "input": "Greet Ada.",
                                "expected_output": "Hello, Ada!",
                                "criteria": ["Uses the name Ada"],
                            }
                        ],
                    },
                    "temperature": 0.35,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "dataset_name": "Greeting quality",
        "results": [
            {
                "input": "Greet Ada.",
                "expected_output": "Hello, Ada!",
                "actual_output": "Actual output at 0.35: Greet Ada.",
                "criteria": ["Uses the name Ada"],
                "passed": False,
            }
        ],
    }
