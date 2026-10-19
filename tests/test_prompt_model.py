from app.models import Base, Prompt, PromptVersion


def test_prompt_model_uses_the_shared_metadata_and_core_columns() -> None:
    table = Prompt.__table__

    assert table.metadata is Base.metadata
    assert table.name == "prompts"
    assert set(table.columns.keys()) == {
        "id",
        "title",
        "content",
        "description",
        "category",
        "is_favorite",
        "is_template",
        "created_at",
        "updated_at",
    }
    assert table.c.id.primary_key is True
    assert table.c.description.nullable is True
    assert table.c.title.nullable is False
    assert table.c.content.nullable is False
    assert table.c.category.server_default is not None
    assert table.c.is_favorite.server_default is not None
    assert table.c.is_template.server_default is not None
    assert table.c.created_at.server_default is not None
    assert table.c.updated_at.server_default is not None


def test_prompt_version_model_has_an_immutable_snapshot_relationship() -> None:
    table = PromptVersion.__table__

    assert table.metadata is Base.metadata
    assert set(table.columns.keys()) == {
        "id",
        "prompt_id",
        "version",
        "content",
        "created_at",
    }
    assert table.c.prompt_id.foreign_keys
    assert next(iter(table.c.prompt_id.foreign_keys)).ondelete == "CASCADE"
    assert Prompt.versions.property.mapper.class_ is PromptVersion
    assert PromptVersion.prompt.property.mapper.class_ is Prompt
    assert {constraint.name for constraint in table.constraints} >= {
        "ck_prompt_versions_version_positive",
        "uq_prompt_versions_prompt_id_version",
    }


def test_prompt_model_declares_application_defaults() -> None:
    table = Prompt.__table__

    assert table.c.id.default is not None
    assert table.c.id.default.is_callable is True
    assert table.c.category.default is not None
    assert table.c.category.default.arg == "general"
    assert table.c.is_favorite.default is not None
    assert table.c.is_favorite.default.arg is False
    assert table.c.is_template.default is not None
    assert table.c.is_template.default.arg is False
    assert table.c.updated_at.onupdate is not None


def test_prompt_model_has_useful_indexes_and_non_blank_constraints() -> None:
    table = Prompt.__table__

    assert {index.name for index in table.indexes} == {
        "ix_prompts_category",
        "ix_prompts_created_at",
    }
    assert {constraint.name for constraint in table.constraints} >= {
        "ck_prompts_title_not_blank",
        "ck_prompts_content_not_blank",
    }
