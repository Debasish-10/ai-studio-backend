from app.llm.provider import LLMProvider
from app.schemas.image import ImageGenerationResponse
from app.services.image_service import ImageService


class StubImageProvider(LLMProvider):
    def __init__(self) -> None:
        self.received_prompt: str | None = None

    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        self.received_prompt = prompt
        return ImageGenerationResponse(image_base64="image-data", mime_type="image/png")

    def generate_text(self, messages, temperature: float = 1.0) -> str:
        return ""

    def stream_text(self, messages, temperature: float = 1.0):
        return iter(())

    def generate_structured_output(self, messages, temperature: float = 1.0):
        raise NotImplementedError


def test_image_service_delegates_to_the_provider() -> None:
    provider = StubImageProvider()
    service = ImageService(provider)

    result = service.generate_image("A red kite")

    assert provider.received_prompt == "A red kite"
    assert result == ImageGenerationResponse(image_base64="image-data", mime_type="image/png")
