from unittest.mock import patch

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.llm.factory import create_llm_provider


def test_factory_creates_gemini_provider() -> None:
    settings = Settings(
        llm_provider="gemini",
        llm_model="test-model",
        gemini_image_model="test-image-model",
        gemini_audio_model="test-audio-model",
        gemini_audio_voice="test-voice",
        gemini_api_key=SecretStr("test-api-key"),
        _env_file=None,
    )

    with patch("app.llm.factory.GeminiProvider") as provider_class:
        provider = create_llm_provider(settings)

    assert provider is provider_class.return_value
    provider_class.assert_called_once_with(
        api_key="test-api-key",
        model="test-model",
        image_model="test-image-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
    )


def test_factory_rejects_missing_gemini_api_key() -> None:
    settings = Settings(llm_provider="gemini", _env_file=None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_llm_provider(settings)


def test_factory_creates_groq_provider() -> None:
    settings = Settings(
        llm_provider="groq",
        groq_model="test-model",
        groq_audio_model="test-audio-model",
        groq_audio_voice="test-voice",
        groq_api_key=SecretStr("test-api-key"),
        _env_file=None,
    )

    with patch("app.llm.factory.GroqProvider") as provider_class:
        provider = create_llm_provider(settings)

    assert provider is provider_class.return_value
    provider_class.assert_called_once_with(
        api_key="test-api-key",
        model="test-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
    )


def test_factory_rejects_missing_groq_api_key() -> None:
    settings = Settings(llm_provider="groq", _env_file=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_llm_provider(settings)


def test_factory_rejects_unknown_provider() -> None:
    settings = Settings(llm_provider="unsupported", _env_file=None)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider(settings)
