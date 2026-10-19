import logging
import base64
from collections.abc import Iterator, Sequence
from typing import Any

from google import genai
from google.genai import types

from app.llm.provider import (
    LLMProvider,
    LLMProviderError,
    validate_structured_chat_output,
)
from app.schemas.chat import ChatRole, ConversationMessage, StructuredChatResponse
from app.schemas.audio import AudioGenerationResponse
from app.schemas.image import ImageGenerationResponse
from app.schemas.tool import ToolCall, ToolDefinition, ToolModelResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini implementation of the application's LLM provider contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        image_model: str | None = None,
        audio_model: str | None = None,
        audio_voice: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be configured for the Gemini provider.")

        self._model = model
        self._image_model = image_model
        self._audio_model = audio_model
        self._audio_voice = audio_voice
        self._client = client or genai.Client(api_key=api_key)

    def generate_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> str:
        contents = [
            types.Content(
                role="model" if message.role == ChatRole.ASSISTANT else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                ),
            )
            text = response.text
        except Exception as error:
            logger.exception("Gemini text generation failed.")
            raise LLMProviderError("Text generation failed.") from error

        if not text:
            raise LLMProviderError("Text generation failed.")

        return text

    def stream_text(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Provide the streaming contract without changing Gemini generation behavior."""

        yield self.generate_text(messages, temperature=temperature)

    def generate_structured_output(
        self,
        messages: Sequence[ConversationMessage],
        temperature: float = 1.0,
    ) -> StructuredChatResponse:
        contents = [
            types.Content(
                role="model" if message.role == ChatRole.ASSISTANT else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=StructuredChatResponse,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
        except Exception as error:
            logger.exception("Gemini structured text generation failed.")
            raise LLMProviderError("Structured text generation failed.") from error

        return validate_structured_chat_output(response.text)

    def request_tool_response(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float = 1.0,
    ) -> ToolModelResponse:
        contents = self._to_contents(messages)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    tools=self._to_gemini_tools(tools),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            function_calls = response.function_calls or []
            if function_calls:
                if len(function_calls) != 1:
                    raise ValueError("Only one tool call is supported by the demo.")
                function_call = function_calls[0]
                return ToolModelResponse(
                    tool_call=ToolCall(
                        id=function_call.id or "gemini-tool-call",
                        name=function_call.name,
                        arguments=function_call.args or {},
                    )
                )
            if not response.text:
                raise ValueError("Gemini returned neither text nor a tool call.")
            return ToolModelResponse(text=response.text)
        except Exception as error:
            logger.exception("Gemini tool request failed.")
            raise LLMProviderError("Tool generation failed.") from error

    def generate_text_from_tool_result(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        tool_call: ToolCall,
        tool_result: str,
        temperature: float = 1.0,
    ) -> str:
        contents = [
            *self._to_contents(messages),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            args=tool_call.arguments,
                        )
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=tool_call.id,
                            name=tool_call.name,
                            response={"output": tool_result},
                        )
                    )
                ],
            ),
        ]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    tools=self._to_gemini_tools(tools),
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="NONE")
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            text = response.text
        except Exception as error:
            logger.exception("Gemini tool result generation failed.")
            raise LLMProviderError("Tool generation failed.") from error

        if not text:
            raise LLMProviderError("Tool generation failed.")
        return text

    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        if not self._image_model:
            raise LLMProviderError("Image generation is not configured.")

        try:
            response = self._client.models.generate_images(
                model=self._image_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
            generated_images = response.generated_images or []
            image = generated_images[0].image if generated_images else None
            if image is None or not image.image_bytes:
                raise ValueError("Gemini returned no image bytes.")
            return ImageGenerationResponse(
                image_base64=base64.b64encode(image.image_bytes).decode("ascii"),
                mime_type=image.mime_type or "image/png",
            )
        except Exception as error:
            logger.exception("Gemini image generation failed.")
            raise LLMProviderError("Image generation failed.") from error

    def generate_audio(self, text: str) -> AudioGenerationResponse:
        if not self._audio_model or not self._audio_voice:
            raise LLMProviderError("Audio generation is not configured.")

        try:
            response = self._client.models.generate_content(
                model=self._audio_model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._audio_voice,
                            )
                        )
                    ),
                ),
            )
            audio = next(
                (
                    part.inline_data
                    for candidate in response.candidates or []
                    for part in candidate.content.parts or []
                    if part.inline_data and part.inline_data.data
                ),
                None,
            )
            if audio is None:
                raise ValueError("Gemini returned no audio bytes.")
            return AudioGenerationResponse(
                audio_base64=base64.b64encode(audio.data).decode("ascii"),
                mime_type=audio.mime_type or "audio/wav",
            )
        except Exception as error:
            logger.exception("Gemini audio generation failed.")
            raise LLMProviderError("Audio generation failed.") from error

    @staticmethod
    def _to_contents(messages: Sequence[ConversationMessage]) -> list[types.Content]:
        return [
            types.Content(
                role="model" if message.role == ChatRole.ASSISTANT else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

    @staticmethod
    def _to_gemini_tools(tools: Sequence[ToolDefinition]) -> list[types.Tool]:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.parameters,
                    )
                    for tool in tools
                ]
            )
        ]
