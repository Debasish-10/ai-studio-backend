import base64

from fastapi.testclient import TestClient

from app.main import app
from app.llm.provider import LLMProviderError
from app.schemas.image import ImageGenerationResponse
from app.services.image_service import get_image_service


class FakeImageService:
    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        return ImageGenerationResponse(
            image_base64=base64.b64encode(prompt.encode()).decode(),
            mime_type="image/png",
        )


class FailingImageService:
    def generate_image(self, prompt: str) -> ImageGenerationResponse:
        raise LLMProviderError("SDK failure: sensitive detail")


def test_generate_image_returns_a_provider_neutral_image_result() -> None:
    app.dependency_overrides[get_image_service] = lambda: FakeImageService()

    try:
        with TestClient(app) as client:
            response = client.post("/images/generate", json={"prompt": "A red kite"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "image_base64": "QSByZWQga2l0ZQ==",
        "mime_type": "image/png",
    }


def test_generate_image_rejects_an_empty_prompt() -> None:
    with TestClient(app) as client:
        response = client.post("/images/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_generate_image_returns_a_safe_error_when_the_provider_fails() -> None:
    app.dependency_overrides[get_image_service] = lambda: FailingImageService()

    try:
        with TestClient(app) as client:
            response = client.post("/images/generate", json={"prompt": "A red kite"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The language model is temporarily unavailable. Please try again later."
    }
    assert "sensitive detail" not in response.text
