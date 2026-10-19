import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.core.database import create_database_engine, get_database_url, normalize_database_url


def test_normalize_database_url_uses_psycopg_for_neon_urls() -> None:
    assert normalize_database_url("postgresql://user:password@host/neondb?sslmode=require") == (
        "postgresql+psycopg://user:password@host/neondb?sslmode=require"
    )


def test_create_database_engine_does_not_connect_during_setup() -> None:
    engine = create_database_engine("postgresql://user:password@host/neondb?sslmode=require")

    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.pool._pre_ping is True


def test_get_database_url_uses_configured_setting_and_psycopg_dialect(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "database_url",
        SecretStr("postgresql://user:password@host/neondb?sslmode=require"),
    )

    assert get_database_url() == "postgresql+psycopg://user:password@host/neondb?sslmode=require"


def test_get_database_url_requires_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_url", None)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be configured"):
        get_database_url()
