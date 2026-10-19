from app.core.config import Settings
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.provider import LLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider for use by application services."""

    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY must be configured for the Gemini provider.")

        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.llm_model,
            image_model=settings.gemini_image_model,
            audio_model=settings.gemini_audio_model,
            audio_voice=settings.gemini_audio_voice,
        )

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY must be configured for the Groq provider.")

        return GroqProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
            audio_model=settings.groq_audio_model,
            audio_voice=settings.groq_audio_voice,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
