from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    frontend_origin: str = "http://localhost:5173"
    database_url: SecretStr | None = None
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    gemini_api_key: SecretStr | None = None
    gemini_image_model: str = "gemini-2.5-flash-image"
    gemini_audio_model: str = "gemini-3.1-flash-tts-preview"
    gemini_audio_voice: str = "Kore"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_audio_model: str = "canopylabs/orpheus-v1-english"
    groq_audio_voice: str = "troy"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
