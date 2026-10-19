from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.llm.provider import LLMProvider, LLMProviderError, validate_structured_chat_output
from app.schemas.chat import ConversationMessage, StructuredChatResponse
from app.schemas.tool import ToolCall, ToolModelResponse
from app.services.chat_service import ChatService, get_chat_service



class FakeLLMProvider(LLMProvider):
    def generate_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        return f"Generated response: {messages[-1].content} at {temperature}"

    def stream_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        return iter(["Generated stream: ", f"{messages[-1].content} at {temperature}"])

    def generate_structured_output(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        return StructuredChatResponse(
            answer=f"Structured response: {messages[-1].content} at {temperature}"
        )


class FailingLLMProvider(LLMProvider):
    def generate_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        raise LLMProviderError("SDK failure: sensitive detail")

    def stream_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        raise LLMProviderError("SDK failure: sensitive detail")

    def generate_structured_output(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        raise LLMProviderError("SDK failure: sensitive detail")


class InvalidStructuredLLMProvider(FakeLLMProvider):
    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output

    def generate_structured_output(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        return validate_structured_chat_output(self._raw_output)


class ToolCallingLLMProvider(FakeLLMProvider):
    def __init__(self, tool_call: ToolCall) -> None:
        self._tool_call = tool_call

    def request_tool_response(self, messages, tools, temperature: float = 1.0) -> ToolModelResponse:
        return ToolModelResponse(tool_call=self._tool_call)

    def generate_text_from_tool_result(
        self,
        messages,
        tools,
        tool_call: ToolCall,
        tool_result: str,
        temperature: float = 1.0,
    ) -> str:
        return f"Final tool response: {tool_result}"


class ToolFailingLLMProvider(FakeLLMProvider):
    def request_tool_response(self, messages, tools, temperature: float = 1.0) -> ToolModelResponse:
        raise LLMProviderError("SDK failure: sensitive detail")


class TextToolLLMProvider(FakeLLMProvider):
    def request_tool_response(self, messages, tools, temperature: float = 1.0) -> ToolModelResponse:
        return ToolModelResponse(text="No tool was needed.")


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeLLMProvider())

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_chat_returns_generated_response(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.json() == {"message": "Generated response: Hello at 1.0"}


def test_chat_stream_returns_plain_text_chunks(client: TestClient) -> None:
    response = client.post("/chat/stream", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Generated stream: Hello at 1.0"


def test_structured_chat_returns_validated_response(client: TestClient) -> None:
    response = client.post("/chat/structured", json={"message": "Hello", "temperature": 0.35})

    assert response.status_code == 200
    assert response.json() == {"answer": "Structured response: Hello at 0.35"}


def test_formatted_chat_returns_a_validated_json_response(client: TestClient) -> None:
    response = client.post(
        "/chat/formatted",
        json={"message": "Hello", "output_format": "json", "temperature": 0.35},
    )

    assert response.status_code == 200
    assert response.json() == {
        "output_format": "json",
        "content": {"answer": "Structured response: Hello at 0.35"},
    }


@pytest.mark.parametrize(
    ("output_format", "instruction"),
    [
        ("markdown", "Respond using Markdown."),
        ("table", "Respond with a Markdown table."),
        (
            "email",
            "Respond as a complete email with a subject, greeting, body, and closing.",
        ),
        ("html", "Respond with valid HTML only."),
    ],
)
def test_formatted_chat_returns_text_for_each_text_format(
    client: TestClient,
    output_format: str,
    instruction: str,
) -> None:
    response = client.post(
        "/chat/formatted",
        json={"message": "Hello", "output_format": output_format},
    )

    assert response.status_code == 200
    assert response.json() == {
        "output_format": output_format,
        "content": f"Generated response: Hello\n\n{instruction} at 1.0",
    }


def test_formatted_chat_rejects_an_unknown_output_format(client: TestClient) -> None:
    response = client.post(
        "/chat/formatted",
        json={"message": "Hello", "output_format": "xml"},
    )

    assert response.status_code == 422


def test_formatted_chat_returns_a_safe_error_when_the_llm_fails() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FailingLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/chat/formatted",
                json={"message": "Hello", "output_format": "markdown"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text


def test_tool_chat_executes_the_requested_tool_and_returns_the_final_response() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        ToolCallingLLMProvider(
            ToolCall(id="call-1", name="add_numbers", arguments={"left": 2, "right": 3})
        )
    )

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/tools", json={"message": "What is 2 plus 3?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "message": "Final tool response: 5",
        "tool_name": "add_numbers",
    }


def test_tool_chat_returns_direct_provider_text_when_no_tool_is_requested() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(TextToolLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/tools", json={"message": "Say hello."})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"message": "No tool was needed.", "tool_name": None}


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(id="call-1", name="unknown_tool", arguments={}),
        ToolCall(id="call-2", name="add_numbers", arguments={"left": "two"}),
    ],
    ids=["unknown-tool", "invalid-arguments"],
)
def test_tool_chat_returns_a_safe_error_for_unknown_or_invalid_tool_calls(
    tool_call: ToolCall,
) -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        ToolCallingLLMProvider(tool_call)
    )

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/tools", json={"message": "Use a tool."})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }


def test_tool_chat_returns_a_safe_error_when_the_provider_fails() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(ToolFailingLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/tools", json={"message": "Use a tool."})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text


def test_chat_stream_accepts_conversation_history(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={
            "message": "What is the capital?",
            "history": [
                {"role": "user", "content": "Let's talk about France."},
                {"role": "assistant", "content": "Sure."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.text == "Generated stream: What is the capital? at 1.0"


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_accepts_conversation_history(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "What is the capital?",
            "history": [
                {"role": "user", "content": "Let's talk about France."},
                {"role": "assistant", "content": "Sure."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Generated response: What is the capital? at 1.0"
    }


def test_chat_passes_temperature_to_the_provider(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hello", "temperature": 0.35})

    assert response.status_code == 200
    assert response.json() == {"message": "Generated response: Hello at 0.35"}


def test_chat_stream_passes_temperature_to_the_provider(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={"message": "Hello", "temperature": 0.35},
    )

    assert response.status_code == 200
    assert response.text == "Generated stream: Hello at 0.35"


def test_chat_rejects_an_invalid_temperature(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hello", "temperature": 2.1})

    assert response.status_code == 422


def test_chat_returns_a_safe_error_when_the_llm_fails() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FailingLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat", json={"message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text


def test_structured_chat_returns_a_safe_error_when_the_llm_fails() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FailingLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/structured", json={"message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"answer":',
        "{}",
        '{"answer": 42}',
        '{"answer": "Generated text", "unexpected": "value"}',
    ],
    ids=["malformed-json", "missing-required-field", "wrong-field-type", "extra-field"],
)
def test_structured_chat_returns_a_safe_error_for_invalid_provider_output(
    raw_output: str,
) -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        InvalidStructuredLLMProvider(raw_output)
    )

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/structured", json={"message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }


def test_chat_stream_returns_a_safe_error_before_streaming_starts() -> None:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FailingLLMProvider())

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/chat/stream", json={"message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text
