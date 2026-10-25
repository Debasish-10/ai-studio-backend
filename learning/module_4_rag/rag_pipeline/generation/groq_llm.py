
import os

from groq import Groq, APIError, APITimeoutError, APIConnectionError

from rag_pipeline.models import (
    ChatMessage,
    LLMGenerationResult
)
from rag_pipeline.generation.llm_interface import LLMService

from typing import Iterator



class LLMGenerationError(Exception):
    pass


class GroqLLMService(LLMService):

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 500
    ):
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

        # Reuse the HTTP client across requests.
        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
            timeout=30.0,
            max_retries=2
        )

        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def generate(
        self,
        messages: list[ChatMessage]
    ) -> LLMGenerationResult:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content
                    }
                    for message in messages
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            if choice.finish_reason == "length":
                raise LLMGenerationError(
                    "LLM output was truncated."
                )

            if not content.strip():
                raise LLMGenerationError(
                    "LLM returned an empty response."
                )

            usage = response.usage

            return LLMGenerationResult(
                text=content,
                model=response.model,
                input_tokens=(
                    usage.prompt_tokens if usage else None
                ),
                output_tokens=(
                    usage.completion_tokens if usage else None
                )
            )

        except (
            APITimeoutError,
            APIConnectionError,
            APIError
        ) as exc:
            raise LLMGenerationError(
                "Groq generation request failed."
            ) from exc

    def stream(
        self,
        messages: list[ChatMessage]
    ) -> Iterator[str]:

        try:
            response_stream = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": message.role,
                            "content": message.content
                        }
                        for message in messages
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=(
                        self.max_completion_tokens
                    ),
                    stream=True
                )
            )

            for chunk in response_stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason == "length":
                    raise LLMGenerationError(
                        "Streamed response was truncated."
                    )

                content = choice.delta.content

                if content:
                    yield content

        except (
            APITimeoutError,
            APIConnectionError,
            APIError
        ) as exc:
            raise LLMGenerationError(
                "Groq streaming request failed."
            ) from exc