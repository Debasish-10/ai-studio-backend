from app.core.config import settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.schemas.audio import AudioGenerationResponse


class AudioService:
    """Application service for provider-agnostic text-to-speech generation."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate_audio(self, text: str) -> AudioGenerationResponse:
        return self._provider.generate_audio(text)


def get_audio_service() -> AudioService:
    """Create the audio service using the configured LLM provider."""

    return AudioService(create_llm_provider(settings))
