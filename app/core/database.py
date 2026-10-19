from functools import lru_cache
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def normalize_database_url(database_url: str) -> str:
    """Use psycopg for standard PostgreSQL and Neon connection strings."""

    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def create_database_engine(database_url: str) -> Engine:
    return create_engine(normalize_database_url(database_url), pool_pre_ping=True)


def get_database_url() -> str:
    """Return the configured database URL using SQLAlchemy's psycopg dialect."""

    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL must be configured before using the database.")
    return normalize_database_url(settings.database_url.get_secret_value())


@lru_cache
def get_database_engine() -> Engine:
    return create_database_engine(get_database_url())


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_database_engine(), autoflush=False, autocommit=False)


def get_db_session() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session per dependent request."""

    with get_session_factory()() as session:
        yield session
