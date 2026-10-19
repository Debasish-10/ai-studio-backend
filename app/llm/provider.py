from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from pydantic import ValidationError

from app.schemas.chat import ConversationMessage, StructuredChatResponse
from app.schemas.audio import AudioGenerationResponse
from app.schemas.image import ImageGenerationResponse
from app.schemas.tool import ToolCall, ToolDefinition, ToolModelResponse


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot return generated text."""


def validate_structured_chat_output(raw_output: str | None) -> StructuredChatResponse:
    """Validate the single structured response contract returned by a provider."""

    try:
        return StructuredChatResponse.model_validate_json(raw_output)
    except (TypeError, ValidationError) as error:
        raise LLMProviderError("Structured text generation failed.") from error


class LLMProvider(ABC):
    """Contract implemented by LLM providers used by application services."""

    @abstractmethod
    def generate_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        """Generate text from an ordered conversation without streaming."""

    @abstractmethod
    def stream_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Generate text chunks from an ordered conversation."""

    @abstractmethod
    def generate_structured_output(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        """Generate and validate the application's structured chat response."""

    def request_tool_response(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float = 1.0,
    ) -> ToolModelResponse:
        """Return either generated text or one provider-translated tool request."""

        raise LLMProviderError("Tool calling is not supported by this provider.")

    def generate_text_from_tool_result(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        tool_call: ToolCall,
        tool_result: str,
        temperature: float = 1.0,
    ) -> str:
        """Generate final text after the application supplies a tool result."""

        raise LLMProviderError("Tool calling is not supported by this provider.")

    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        """Generate one image from a prompt when the provider supports it."""

        raise LLMProviderError("Image generation is not supported by this provider.")

    def generate_audio(self, text: str) -> AudioGenerationResponse:
        """Generate speech audio from text when the provider supports it."""

        raise LLMProviderError("Audio generation is not supported by this provider.")
