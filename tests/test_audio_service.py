from app.llm.provider import LLMProvider
from app.schemas.audio import AudioGenerationResponse
from app.services.audio_service import AudioService


class StubAudioProvider(LLMProvider):
    def __init__(self) -> None:
        self.received_text: str | None = None

    def generate_audio(self, text: str) -> AudioGenerationResponse:
        self.received_text = text
        return AudioGenerationResponse(audio_base64="audio-data", mime_type="audio/wav")

    def generate_text(self, messages, temperature: float = 1.0) -> str:
        return ""

    def stream_text(self, messages, temperature: float = 1.0):
        return iter(())

    def generate_structured_output(self, messages, temperature: float = 1.0):
        raise NotImplementedError


def test_audio_service_delegates_to_the_provider() -> None:
    provider = StubAudioProvider()
    service = AudioService(provider)

    result = service.generate_audio("Hello, world.")

    assert provider.received_text == "Hello, world."
    assert result == AudioGenerationResponse(audio_base64="audio-data", mime_type="audio/wav")
