from collections.abc import Iterator

from app.core.config import settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.schemas.chat import (
    ChatResponse,
    ChatRole,
    ConversationMessage,
    FormattedChatResponse,
    OutputFormat,
    StructuredChatResponse,
)
from app.schemas.tool import ToolChatResponse
from app.tools import execute_demo_tool, get_demo_tools


_TEXT_OUTPUT_FORMAT_INSTRUCTIONS = {
    OutputFormat.MARKDOWN: "Respond using Markdown.",
    OutputFormat.TABLE: "Respond with a Markdown table.",
    OutputFormat.EMAIL: "Respond as a complete email with a subject, greeting, body, and closing.",
    OutputFormat.HTML: "Respond with valid HTML only.",
}


class ChatService:
    """Application service for one-shot chat responses."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def create_response(
        self,
        message: str,
        history: list[ConversationMessage] | None = None,
        temperature: float = 1.0,
    ) -> ChatResponse:
        conversation = [
            *(history or []),
            ConversationMessage(role=ChatRole.USER, content=message),
        ]
        generated_text = self._provider.generate_text(conversation, temperature=temperature)
        return ChatResponse(message=generated_text)

    def stream_response(
        self,
        message: str,
        history: list[ConversationMessage] | None = None,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream generated text for the current message and conversation history."""

        conversation = [
            *(history or []),
            ConversationMessage(role=ChatRole.USER, content=message),
        ]
        yield from self._provider.stream_text(conversation, temperature=temperature)

    def create_structured_response(
        self,
        message: str,
        history: list[ConversationMessage] | None = None,
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        """Generate a validated structured response for the conversation."""

        conversation = [
            *(history or []),
            ConversationMessage(role=ChatRole.USER, content=message),
        ]
        return self._provider.generate_structured_output(
            conversation,
            temperature=temperature,
        )

    def create_formatted_response(
        self,
        message: str,
        output_format: OutputFormat,
        history: list[ConversationMessage] | None = None,
        temperature: float = 1.0,
    ) -> FormattedChatResponse:
        """Generate JSON or a requested text-based output format."""

        if output_format == OutputFormat.JSON:
            return FormattedChatResponse(
                output_format=output_format,
                content=self.create_structured_response(message, history, temperature),
            )

        conversation = [
            *(history or []),
            ConversationMessage(
                role=ChatRole.USER,
                content=f"{message}\n\n{_TEXT_OUTPUT_FORMAT_INSTRUCTIONS[output_format]}",
            ),
        ]
        generated_text = self._provider.generate_text(conversation, temperature=temperature)
        return FormattedChatResponse(
            output_format=output_format,
            content=generated_text,
        )

    def create_tool_response(
        self,
        message: str,
        history: list[ConversationMessage] | None = None,
        temperature: float = 1.0,
    ) -> ToolChatResponse:
        """Run one provider-requested demo tool and return the final model response."""

        conversation = [
            *(history or []),
            ConversationMessage(role=ChatRole.USER, content=message),
        ]
        tools = get_demo_tools()
        model_response = self._provider.request_tool_response(
            conversation,
            tools,
            temperature=temperature,
        )
        if model_response.text is not None:
            return ToolChatResponse(message=model_response.text)

        tool_call = model_response.tool_call
        if tool_call is None:
            raise RuntimeError("Provider returned an invalid tool response.")
        tool_result = execute_demo_tool(tool_call)
        final_text = self._provider.generate_text_from_tool_result(
            conversation,
            tools,
            tool_call,
            tool_result,
            temperature=temperature,
        )
        return ToolChatResponse(message=final_text, tool_name=tool_call.name)


def get_chat_service() -> ChatService:
    """Create the chat service using the configured LLM provider."""

    return ChatService(create_llm_provider(settings))
