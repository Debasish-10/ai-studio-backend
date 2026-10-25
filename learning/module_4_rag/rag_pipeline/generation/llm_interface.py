
from abc import ABC, abstractmethod
from typing import Iterator

from rag_pipeline.models import (
    ChatMessage,
    LLMGenerationResult
)


class LLMService(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage]
    ) -> LLMGenerationResult:
        pass

    def stream(
        self,
        messages: list[ChatMessage]
    ) -> Iterator[str]:
        raise NotImplementedError(
            "This provider does not support streaming."
        )
