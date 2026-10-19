import logging
import json
import base64
from collections.abc import Iterator, Sequence
from typing import Any

from groq import Groq

from app.llm.provider import (
    LLMProvider,
    LLMProviderError,
    validate_structured_chat_output,
)
from app.schemas.chat import ConversationMessage, StructuredChatResponse
from app.schemas.audio import AudioGenerationResponse
from app.schemas.image import ImageGenerationResponse
from app.schemas.tool import ToolCall, ToolDefinition, ToolModelResponse

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq implementation of the application's LLM provider contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        audio_model: str | None = None,
        audio_voice: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY must be configured for the Groq provider.")

        self._model = model
        self._audio_model = audio_model
        self._audio_voice = audio_voice
        self._client = client or Groq(api_key=api_key)

    def generate_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        groq_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content
        except Exception as error:
            logger.exception("Groq text generation failed.")
            raise LLMProviderError("Text generation failed.") from error

        if not text:
            raise LLMProviderError("Text generation failed.")

        return text

    def stream_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        groq_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as error:
            logger.exception("Groq text streaming failed.")
            raise LLMProviderError("Text generation failed.") from error

    def generate_structured_output(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        groq_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_chat_response",
                        "strict": True,
                        "schema": StructuredChatResponse.model_json_schema(),
                    },
                },
            )
        except Exception as error:
            logger.exception("Groq structured text generation failed.")
            raise LLMProviderError("Structured text generation failed.") from error

        return validate_structured_chat_output(response.choices[0].message.content)

    def request_tool_response(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float = 1.0,
    ) -> ToolModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._to_groq_messages(messages),
                tools=self._to_groq_tools(tools),
                tool_choice="auto",
                temperature=temperature,
            )
            message = response.choices[0].message
            if message.tool_calls:
                if len(message.tool_calls) != 1:
                    raise ValueError("Only one tool call is supported by the demo.")
                tool_call = message.tool_calls[0]
                return ToolModelResponse(
                    tool_call=ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    )
                )
            if not message.content:
                raise ValueError("Groq returned neither text nor a tool call.")
            return ToolModelResponse(text=message.content)
        except Exception as error:
            logger.exception("Groq tool request failed.")
            raise LLMProviderError("Tool generation failed.") from error

    def generate_text_from_tool_result(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        tool_call: ToolCall,
        tool_result: str,
        temperature: float = 1.0,
    ) -> str:
        groq_messages = [
            *self._to_groq_messages(messages),
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": tool_result,
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                tools=self._to_groq_tools(tools),
                tool_choice="none",
                temperature=temperature,
            )
            text = response.choices[0].message.content
        except Exception as error:
            logger.exception("Groq tool result generation failed.")
            raise LLMProviderError("Tool generation failed.") from error

        if not text:
            raise LLMProviderError("Tool generation failed.")
        return text

    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        """Groq currently supports image understanding, not image generation."""

        raise LLMProviderError("Image generation is not supported by the Groq provider.")

    def generate_audio(self, text: str) -> AudioGenerationResponse:
        if not self._audio_model or not self._audio_voice:
            raise LLMProviderError("Audio generation is not configured.")

        try:
            response = self._client.audio.speech.create(
                model=self._audio_model,
                voice=self._audio_voice,
                input=text,
                response_format="wav",
            )
            audio_bytes = response.read()
            if not audio_bytes:
                raise ValueError("Groq returned no audio bytes.")
            return AudioGenerationResponse(
                audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
                mime_type="audio/wav",
            )
        except Exception as error:
            logger.exception("Groq audio generation failed.")
            raise LLMProviderError("Audio generation failed.") from error

    @staticmethod
    def _to_groq_messages(messages: Sequence[ConversationMessage]) -> list[dict[str, str]]:
        return [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

    @staticmethod
    def _to_groq_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(),
            }
            for tool in tools
        ]
