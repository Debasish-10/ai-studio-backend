from pydantic import SecretStr

from app.core.config import Settings


def test_llm_settings_have_gemini_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.database_url is None
    assert settings.gemini_api_key is None
    assert settings.groq_api_key is None
    assert settings.groq_model == "openai/gpt-oss-20b"


def test_llm_settings_load_environment_values(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://database.example/test")
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key")
    monkeypatch.setenv("GROQ_MODEL", "test-groq-model")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "test-provider"
    assert settings.llm_model == "test-model"
    assert isinstance(settings.database_url, SecretStr)
    assert settings.database_url.get_secret_value() == "postgresql://database.example/test"
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert settings.gemini_api_key.get_secret_value() == "test-api-key"
    assert isinstance(settings.groq_api_key, SecretStr)
    assert settings.groq_api_key.get_secret_value() == "test-groq-api-key"
    assert settings.groq_model == "test-groq-model"
