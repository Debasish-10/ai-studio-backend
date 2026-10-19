from app.core.config import settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.schemas.image import ImageGenerationResponse


class ImageService:
    """Application service for provider-agnostic image generation."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        return self._provider.generate_image(prompt)


def get_image_service() -> ImageService:
    """Create the image service using the configured LLM provider."""

    return ImageService(create_llm_provider(settings))
