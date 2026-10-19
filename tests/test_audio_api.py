import base64

from fastapi.testclient import TestClient

from app.main import app
from app.llm.provider import LLMProviderError
from app.schemas.audio import AudioGenerationResponse
from app.services.audio_service import get_audio_service


class FakeAudioService:
    def generate_audio(self, text: str) -> AudioGenerationResponse:
        return AudioGenerationResponse(
            audio_base64=base64.b64encode(text.encode()).decode(),
            mime_type="audio/wav",
        )


class FailingAudioService:
    def generate_audio(self, text: str) -> AudioGenerationResponse:
        raise LLMProviderError("SDK failure: sensitive detail")


def test_generate_audio_returns_a_provider_neutral_audio_result() -> None:
    app.dependency_overrides[get_audio_service] = lambda: FakeAudioService()

    try:
        with TestClient(app) as client:
            response = client.post("/audio/generate", json={"text": "Hello, world."})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "audio_base64": "SGVsbG8sIHdvcmxkLg==",
        "mime_type": "audio/wav",
    }


def test_generate_audio_rejects_empty_text() -> None:
    with TestClient(app) as client:
        response = client.post("/audio/generate", json={"text": ""})

    assert response.status_code == 422


def test_generate_audio_returns_a_safe_error_when_the_provider_fails() -> None:
    app.dependency_overrides[get_audio_service] = lambda: FailingAudioService()

    try:
        with TestClient(app) as client:
            response = client.post("/audio/generate", json={"text": "Hello, world."})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text
