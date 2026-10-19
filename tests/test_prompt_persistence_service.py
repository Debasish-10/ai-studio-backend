from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.models import Prompt, PromptVersion
from app.repositories import PromptRepository, PromptVersionRepository
from app.schemas.prompt import PromptCreate, PromptListQuery, PromptUpdate
from app.services.prompt_service import PromptService


def build_service() -> tuple[PromptService, Mock, Mock, Mock]:
    session = Mock(spec=Session)
    repository = Mock(spec=PromptRepository)
    version_repository = Mock(spec=PromptVersionRepository)
    repository.session = session
    return PromptService(repository, version_repository), repository, version_repository, session


def test_service_creates_prompt_and_commits_the_repository_mutation() -> None:
    service, repository, version_repository, session = build_service()

    result = service.create_prompt(PromptCreate(title="Greeting", content="Say hello"))

    created_prompt = repository.create.call_args.args[0]
    assert result is created_prompt
    assert created_prompt.title == "Greeting"
    assert created_prompt.category == "general"
    snapshot = version_repository.create.call_args.args[0]
    assert snapshot.version == 1
    assert snapshot.content == "Say hello"
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()


def test_service_reads_and_lists_without_starting_transactions() -> None:
    service, repository, _, session = build_service()
    prompt_id = uuid4()
    now = datetime.now(UTC)
    prompt = Prompt(
        id=uuid4(),
        title="Greeting",
        content="Say hello",
        category="general",
        is_favorite=False,
        is_template=False,
        created_at=now,
        updated_at=now,
    )
    repository.get_by_id.return_value = prompt
    repository.list_prompts.return_value = ([prompt], 21)

    assert service.get_prompt(prompt_id) is prompt
    page = service.list_prompts(PromptListQuery(page=2, page_size=10))

    repository.get_by_id.assert_called_once_with(prompt_id)
    repository.list_prompts.assert_called_once_with(
        search=None,
        category=None,
        is_favorite=None,
        is_template=None,
        page=2,
        page_size=10,
    )
    assert [item.title for item in page.items] == ["Greeting"]
    assert page.total == 21
    assert page.total_pages == 3
    session.commit.assert_not_called()


def test_service_lists_categories_without_committing() -> None:
    service, repository, _, session = build_service()
    repository.list_categories.return_value = ["general", "marketing"]

    assert service.list_categories() == ["general", "marketing"]

    repository.list_categories.assert_called_once_with()
    session.commit.assert_not_called()


def test_service_updates_or_deletes_only_existing_prompts() -> None:
    service, repository, version_repository, session = build_service()
    prompt_id = uuid4()
    prompt = Prompt(id=prompt_id, title="Greeting", content="Say hello")
    repository.get_by_id.return_value = prompt
    repository.update.side_effect = lambda record, values: [setattr(record, key, value) for key, value in values.items()]

    assert service.update_prompt(prompt_id, PromptUpdate(title="Welcome")) is prompt
    assert service.delete_prompt(prompt_id) is True

    repository.update.assert_called_once_with(prompt, {"title": "Welcome"})
    version_repository.next_version_number.assert_not_called()
    version_repository.create.assert_not_called()
    repository.delete.assert_called_once_with(prompt)
    assert session.commit.call_count == 2


def test_service_does_not_commit_not_found_update_or_delete() -> None:
    service, repository, _, session = build_service()
    repository.get_by_id.return_value = None

    assert service.update_prompt(uuid4(), PromptUpdate(title="Welcome")) is None
    assert service.delete_prompt(uuid4()) is False

    repository.update.assert_not_called()
    repository.delete.assert_not_called()
    session.commit.assert_not_called()


def test_service_marks_and_unmarks_favorites_and_templates() -> None:
    service, repository, version_repository, session = build_service()
    prompt_id = uuid4()
    prompt = Prompt(id=prompt_id, title="Greeting", content="Say hello")
    repository.get_by_id.return_value = prompt
    repository.update.side_effect = lambda record, values: [setattr(record, key, value) for key, value in values.items()]

    assert service.set_favorite(prompt_id, True) is prompt
    assert service.set_favorite(prompt_id, False) is prompt
    assert service.set_template(prompt_id, True) is prompt
    assert service.set_template(prompt_id, False) is prompt

    assert repository.update.call_args_list[0].args == (prompt, {"is_favorite": True})
    assert repository.update.call_args_list[1].args == (prompt, {"is_favorite": False})
    assert repository.update.call_args_list[2].args == (prompt, {"is_template": True})
    assert repository.update.call_args_list[3].args == (prompt, {"is_template": False})
    version_repository.next_version_number.assert_not_called()
    version_repository.create.assert_not_called()
    assert session.commit.call_count == 4


def test_service_does_not_version_title_description_or_category_updates() -> None:
    service, repository, version_repository, session = build_service()
    prompt_id = uuid4()
    prompt = Prompt(
        id=prompt_id,
        title="Greeting",
        content="Say hello",
        description="Initial description",
        category="general",
    )
    repository.get_by_id.return_value = prompt
    repository.update.side_effect = lambda record, values: [setattr(record, key, value) for key, value in values.items()]

    service.update_prompt(prompt_id, PromptUpdate(title="Welcome"))
    service.update_prompt(prompt_id, PromptUpdate(description="Friendly greeting"))
    service.update_prompt(prompt_id, PromptUpdate(category="marketing"))

    assert prompt.title == "Welcome"
    assert prompt.description == "Friendly greeting"
    assert prompt.category == "marketing"
    version_repository.next_version_number.assert_not_called()
    version_repository.create.assert_not_called()
    assert session.commit.call_count == 3


def test_service_creates_immutable_snapshots_for_content_updates() -> None:
    service, repository, version_repository, session = build_service()
    prompt_id = uuid4()
    prompt = Prompt(
        id=prompt_id,
        title="Greeting",
        content="Say hello",
        description=None,
        category="general",
        is_favorite=False,
        is_template=False,
    )
    repository.get_by_id.return_value = prompt
    repository.update.side_effect = lambda record, values: [setattr(record, key, value) for key, value in values.items()]
    version_repository.next_version_number.side_effect = [2, 3]

    service.update_prompt(prompt_id, PromptUpdate(title="Welcome"))
    service.update_prompt(prompt_id, PromptUpdate(content="Say hello warmly"))
    second_version = version_repository.create.call_args_list[0].args[0]
    service.update_prompt(prompt_id, PromptUpdate(content="Say hello enthusiastically"))
    third_version = version_repository.create.call_args_list[1].args[0]

    assert second_version.version == 2
    assert second_version.content == "Say hello warmly"
    assert third_version.version == 3
    assert third_version.content == "Say hello enthusiastically"
    assert second_version.content == "Say hello warmly"
    assert session.commit.call_count == 3


def test_service_versions_a_mixed_metadata_and_content_update_once() -> None:
    service, repository, version_repository, _ = build_service()
    prompt_id = uuid4()
    prompt = Prompt(id=prompt_id, title="Greeting", content="Say hello", category="general")
    repository.get_by_id.return_value = prompt
    repository.update.side_effect = lambda record, values: [setattr(record, key, value) for key, value in values.items()]
    version_repository.next_version_number.return_value = 2

    service.update_prompt(
        prompt_id,
        PromptUpdate(title="Welcome", category="marketing", content="Say hello warmly"),
    )

    repository.update.assert_called_once_with(
        prompt,
        {"title": "Welcome", "content": "Say hello warmly", "category": "marketing"},
    )
    snapshot = version_repository.create.call_args.args[0]
    assert snapshot.version == 2
    assert snapshot.content == "Say hello warmly"
    version_repository.next_version_number.assert_called_once_with(prompt_id)


def test_service_skips_versioning_and_commit_for_a_same_content_update() -> None:
    service, repository, version_repository, session = build_service()
    prompt_id = uuid4()
    prompt = Prompt(id=prompt_id, title="Greeting", content="Say hello", category="general")
    repository.get_by_id.return_value = prompt

    assert service.update_prompt(prompt_id, PromptUpdate(content="Say hello")) is prompt

    repository.update.assert_not_called()
    version_repository.next_version_number.assert_not_called()
    version_repository.create.assert_not_called()
    session.commit.assert_not_called()


def test_service_reads_version_history_and_individual_versions() -> None:
    service, repository, version_repository, _ = build_service()
    prompt_id = uuid4()
    now = datetime.now(UTC)
    prompt = Prompt(id=prompt_id, title="Greeting", content="Say hello")
    version = PromptVersion(
        id=uuid4(),
        prompt_id=prompt_id,
        version=2,
        content="Say hello warmly",
        created_at=now,
    )
    repository.get_by_id.return_value = prompt
    version_repository.list_for_prompt.return_value = [version]
    version_repository.get_by_prompt_and_version.return_value = version

    assert [item.version for item in service.list_versions(prompt_id)] == [2]
    assert service.get_version(prompt_id, 2).content == "Say hello warmly"

    version_repository.list_for_prompt.assert_called_once_with(prompt_id)
    version_repository.get_by_prompt_and_version.assert_called_once_with(prompt_id, 2)


def test_service_does_not_query_versions_for_missing_prompt() -> None:
    service, repository, version_repository, _ = build_service()
    repository.get_by_id.return_value = None

    assert service.list_versions(uuid4()) is None
    assert service.get_version(uuid4(), 1) is None

    version_repository.list_for_prompt.assert_not_called()
    version_repository.get_by_prompt_and_version.assert_not_called()


def test_service_rolls_back_when_snapshot_creation_fails() -> None:
    service, _, version_repository, session = build_service()
    version_repository.create.side_effect = RuntimeError("version persistence failed")

    with pytest.raises(RuntimeError, match="version persistence failed"):
        service.create_prompt(PromptCreate(title="Greeting", content="Say hello"))

    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()


def test_service_rolls_back_when_commit_fails() -> None:
    service, _, _, session = build_service()
    session.commit.side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.create_prompt(PromptCreate(title="Greeting", content="Say hello"))

    session.rollback.assert_called_once_with()
