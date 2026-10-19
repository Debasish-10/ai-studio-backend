from collections.abc import Iterator

import pytest

from app.llm.provider import LLMProvider
from app.schemas.chat import (
    ChatRole,
    ConversationMessage,
    OutputFormat,
    StructuredChatResponse,
)
from app.schemas.tool import ToolCall, ToolModelResponse
from app.services.chat_service import ChatService


class StubLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.received_messages: list[ConversationMessage] | None = None
        self.received_temperature: float | None = None
        self.stream_chunks = ["Generated ", "response"]

    def generate_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        self.received_messages = messages
        self.received_temperature = temperature
        return "Generated response"

    def stream_text(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        self.received_messages = messages
        self.received_temperature = temperature
        return iter(self.stream_chunks)

    def generate_structured_output(
        self,
        messages: list[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        self.received_messages = messages
        self.received_temperature = temperature
        return StructuredChatResponse(answer="Structured response")


def test_chat_service_sends_the_first_message_to_its_provider() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    response = service.create_response("Hello")

    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content="Hello")
    ]
    assert response.message == "Generated response"


def test_chat_service_sends_history_before_the_current_user_message() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)
    history = [
        ConversationMessage(role=ChatRole.USER, content="My name is Ada."),
        ConversationMessage(role=ChatRole.ASSISTANT, content="Nice to meet you, Ada."),
    ]

    service.create_response("What is my name?", history)

    assert provider.received_messages == [
        *history,
        ConversationMessage(role=ChatRole.USER, content="What is my name?"),
    ]


def test_chat_service_streams_provider_chunks_for_the_current_message() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    chunks = list(service.stream_response("Hello"))

    assert chunks == ["Generated ", "response"]
    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content="Hello")
    ]


def test_chat_service_streams_history_before_the_current_user_message() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)
    history = [
        ConversationMessage(role=ChatRole.USER, content="My name is Ada."),
        ConversationMessage(role=ChatRole.ASSISTANT, content="Nice to meet you, Ada."),
    ]

    list(service.stream_response("What is my name?", history))

    assert provider.received_messages == [
        *history,
        ConversationMessage(role=ChatRole.USER, content="What is my name?"),
    ]


def test_chat_service_passes_temperature_to_the_provider() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    service.create_response("Hello", temperature=0.35)

    assert provider.received_temperature == 0.35


def test_chat_service_passes_temperature_to_the_streaming_provider() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    list(service.stream_response("Hello", temperature=0.35))

    assert provider.received_temperature == 0.35


def test_chat_service_creates_a_structured_response() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    response = service.create_structured_response("Hello", temperature=0.35)

    assert response == StructuredChatResponse(answer="Structured response")
    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content="Hello")
    ]
    assert provider.received_temperature == 0.35


def test_chat_service_creates_a_json_formatted_response() -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    response = service.create_formatted_response("Hello", OutputFormat.JSON, temperature=0.35)

    assert response.output_format == OutputFormat.JSON
    assert response.content == StructuredChatResponse(answer="Structured response")
    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content="Hello")
    ]
    assert provider.received_temperature == 0.35


@pytest.mark.parametrize(
    ("output_format", "instruction"),
    [
        (OutputFormat.MARKDOWN, "Respond using Markdown."),
        (OutputFormat.TABLE, "Respond with a Markdown table."),
        (
            OutputFormat.EMAIL,
            "Respond as a complete email with a subject, greeting, body, and closing.",
        ),
        (OutputFormat.HTML, "Respond with valid HTML only."),
    ],
)
def test_chat_service_creates_text_formatted_responses(
    output_format: OutputFormat,
    instruction: str,
) -> None:
    provider = StubLLMProvider()
    service = ChatService(provider)

    response = service.create_formatted_response(
        "Hello",
        output_format,
        temperature=0.35,
    )

    assert response.output_format == output_format
    assert response.content == "Generated response"
    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content=f"Hello\n\n{instruction}")
    ]
    assert provider.received_temperature == 0.35


class ToolCallingStubLLMProvider(StubLLMProvider):
    def __init__(self, tool_call: ToolCall) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.received_tools = None
        self.received_tool_result: str | None = None

    def request_tool_response(self, messages, tools, temperature: float = 1.0) -> ToolModelResponse:
        self.received_messages = list(messages)
        self.received_tools = tools
        self.received_temperature = temperature
        return ToolModelResponse(tool_call=self.tool_call)

    def generate_text_from_tool_result(
        self,
        messages,
        tools,
        tool_call: ToolCall,
        tool_result: str,
        temperature: float = 1.0,
    ) -> str:
        self.received_messages = list(messages)
        self.received_tools = tools
        self.received_tool_result = tool_result
        self.received_temperature = temperature
        return f"The tool result is {tool_result}."


def test_chat_service_executes_a_requested_tool_and_returns_the_final_response() -> None:
    provider = ToolCallingStubLLMProvider(
        ToolCall(id="call-1", name="add_numbers", arguments={"left": 2, "right": 3})
    )
    service = ChatService(provider)

    response = service.create_tool_response("What is 2 plus 3?", temperature=0.35)

    assert response.message == "The tool result is 5."
    assert response.tool_name == "add_numbers"
    assert provider.received_tool_result == "5"
    assert [tool.name for tool in provider.received_tools] == [
        "add_numbers",
        "get_project_name",
    ]
    assert provider.received_messages == [
        ConversationMessage(role=ChatRole.USER, content="What is 2 plus 3?")
    ]
    assert provider.received_temperature == 0.35
