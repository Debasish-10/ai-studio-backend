from pathlib import Path

from alembic.config import Config

from app.models import Base, Prompt, PromptVersion


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_alembic_uses_the_standard_migration_structure() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))

    assert config.get_main_option("script_location") == str(PROJECT_ROOT / "alembic")
    assert config.get_main_option("sqlalchemy.url") == ""
    assert (PROJECT_ROOT / "alembic" / "env.py").is_file()
    assert (PROJECT_ROOT / "alembic" / "versions").is_dir()


def test_shared_metadata_includes_prompt_and_version_models() -> None:
    assert Base.metadata.tables == {
        "prompts": Prompt.__table__,
        "prompt_versions": PromptVersion.__table__,
    }
