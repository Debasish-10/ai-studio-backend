from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Iterator
from unittest.mock import Mock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.prompts import get_prompt_service
from app.main import app
from app.models import Prompt, PromptVersion
from app.repositories import PromptRepository
from app.schemas.prompt import (
    PromptCreate,
    PromptListQuery,
    PromptPage,
    PromptRead,
    PromptUpdate,
    PromptVersionRead,
)
from app.services.prompt_service import PromptService


class FakePromptService:
    def __init__(self) -> None:
        self.prompts: dict[UUID, Prompt] = {}
        self.versions: dict[UUID, list[PromptVersion]] = {}

    def create_prompt(self, data: PromptCreate) -> Prompt:
        now = datetime.now(UTC)
        prompt = Prompt(
            id=uuid4(),
            created_at=now,
            updated_at=now,
            **data.model_dump(),
        )
        self.prompts[prompt.id] = prompt
        self.versions[prompt.id] = [self._snapshot(prompt, version=1)]
        return prompt

    def get_prompt(self, prompt_id: UUID) -> Prompt | None:
        return self.prompts.get(prompt_id)

    def list_prompts(self, query: PromptListQuery) -> PromptPage:
        prompts = list(self.prompts.values())
        if query.search:
            search = query.search.casefold()
            prompts = [
                prompt
                for prompt in prompts
                if search
                in " ".join(
                    value for value in [prompt.title, prompt.content, prompt.description] if value
                ).casefold()
            ]
        if query.category:
            prompts = [prompt for prompt in prompts if prompt.category == query.category]
        if query.is_favorite is not None:
            prompts = [prompt for prompt in prompts if prompt.is_favorite == query.is_favorite]
        if query.is_template is not None:
            prompts = [prompt for prompt in prompts if prompt.is_template == query.is_template]

        prompts.sort(key=lambda prompt: (prompt.created_at, prompt.id), reverse=True)
        total = len(prompts)
        start = (query.page - 1) * query.page_size
        page_items = prompts[start : start + query.page_size]
        return PromptPage(
            items=[PromptRead.model_validate(prompt) for prompt in page_items],
            page=query.page,
            page_size=query.page_size,
            total=total,
            total_pages=(total + query.page_size - 1) // query.page_size,
        )

    def list_categories(self) -> list[str]:
        return sorted({prompt.category for prompt in self.prompts.values()})

    def update_prompt(self, prompt_id: UUID, data: PromptUpdate) -> Prompt | None:
        prompt = self.prompts.get(prompt_id)
        if prompt is None:
            return None
        values = data.model_dump(exclude_unset=True)
        if all(getattr(prompt, field) == value for field, value in values.items()):
            return prompt
        content_changed = "content" in values and values["content"] != prompt.content
        for field, value in values.items():
            setattr(prompt, field, value)
        prompt.updated_at = datetime.now(UTC)
        if content_changed:
            self.versions[prompt_id].append(
                self._snapshot(prompt, version=len(self.versions[prompt_id]) + 1)
            )
        return prompt

    def delete_prompt(self, prompt_id: UUID) -> bool:
        deleted = self.prompts.pop(prompt_id, None) is not None
        if deleted:
            self.versions.pop(prompt_id, None)
        return deleted

    def set_favorite(self, prompt_id: UUID, is_favorite: bool) -> Prompt | None:
        return self.update_prompt(prompt_id, PromptUpdate(is_favorite=is_favorite))

    def set_template(self, prompt_id: UUID, is_template: bool) -> Prompt | None:
        return self.update_prompt(prompt_id, PromptUpdate(is_template=is_template))

    def list_versions(self, prompt_id: UUID) -> list[PromptVersionRead] | None:
        versions = self.versions.get(prompt_id)
        if versions is None:
            return None
        return [PromptVersionRead.model_validate(version) for version in reversed(versions)]

    def get_version(self, prompt_id: UUID, version: int) -> PromptVersionRead | None:
        versions = self.versions.get(prompt_id)
        if versions is None:
            return None
        for prompt_version in versions:
            if prompt_version.version == version:
                return PromptVersionRead.model_validate(prompt_version)
        return None

    @staticmethod
    def _snapshot(prompt: Prompt, version: int) -> PromptVersion:
        return PromptVersion(
            id=uuid4(),
            prompt_id=prompt.id,
            version=version,
            content=prompt.content,
            created_at=datetime.now(UTC),
        )


@contextmanager
def prompt_client(service: FakePromptService) -> Iterator[TestClient]:
    app.dependency_overrides[get_prompt_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def create_prompt(
    service: FakePromptService,
    title: str = "Greeting",
    content: str = "Say hello",
    **values: object,
) -> Prompt:
    return service.create_prompt(PromptCreate(title=title, content=content, **values))


def test_prompt_service_dependency_builds_the_repository_from_the_request_session() -> None:
    session = Mock(spec=Session)

    service = get_prompt_service(session)

    assert isinstance(service, PromptService)
    assert isinstance(service.repository, PromptRepository)
    assert service.session is session


def test_create_prompt_returns_201_and_a_prompt_response() -> None:
    with prompt_client(FakePromptService()) as client:
        response = client.post(
            "/prompts",
            json={"title": "Greeting", "content": "Say hello", "is_template": True},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Greeting"
    assert body["content"] == "Say hello"
    assert body["category"] == "general"
    assert body["is_template"] is True
    assert UUID(body["id"])


def test_list_and_get_prompt_return_persisted_prompt_responses() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        list_response = client.get("/prompts")
        get_response = client.get(f"/prompts/{prompt.id}")

    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()["items"]] == [str(prompt.id)]
    assert list_response.json()["total"] == 1
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Greeting"


def test_list_search_is_case_insensitive_across_prompt_text_fields() -> None:
    service = FakePromptService()
    create_prompt(service, title="React guide", content="Build interfaces", description="Frontend notes")
    create_prompt(service, title="Database guide", content="Use postgresql")

    with prompt_client(service) as client:
        title_response = client.get("/prompts?search=REACT")
        content_response = client.get("/prompts?search=POSTGRESQL")
        description_response = client.get("/prompts?search=frontend")
        blank_response = client.get("/prompts?search=%20%20%20")

    assert [item["title"] for item in title_response.json()["items"]] == ["React guide"]
    assert [item["title"] for item in content_response.json()["items"]] == ["Database guide"]
    assert [item["title"] for item in description_response.json()["items"]] == ["React guide"]
    assert blank_response.json()["total"] == 2


def test_list_composes_category_favorite_and_template_filters() -> None:
    service = FakePromptService()
    create_prompt(
        service,
        title="React template",
        content="Reusable component prompt",
        category="coding",
        is_favorite=True,
        is_template=True,
    )
    create_prompt(
        service,
        title="React note",
        content="Regular component prompt",
        category="coding",
        is_favorite=True,
    )
    create_prompt(service, title="React marketing", content="Campaign", category="marketing", is_favorite=True)

    with prompt_client(service) as client:
        response = client.get(
            "/prompts?search=react&category=coding&is_favorite=true&is_template=true"
        )
        favorites_response = client.get("/prompts?is_favorite=true")
        templates_response = client.get("/prompts?is_template=true")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["title"] for item in response.json()["items"]] == ["React template"]
    assert favorites_response.json()["total"] == 3
    assert templates_response.json()["total"] == 1


def test_list_paginates_and_reports_total_metadata() -> None:
    service = FakePromptService()
    create_prompt(service, title="First", content="One")
    create_prompt(service, title="Second", content="Two")
    create_prompt(service, title="Third", content="Three")

    with prompt_client(service) as client:
        response = client.get("/prompts?page=2&page_size=1")

    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 2
    assert body["page_size"] == 1
    assert body["total"] == 3
    assert body["total_pages"] == 3
    assert len(body["items"]) == 1


def test_list_returns_empty_page_metadata_when_nothing_matches() -> None:
    service = FakePromptService()
    create_prompt(service)

    with prompt_client(service) as client:
        response = client.get("/prompts?search=missing")

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "page": 1,
        "page_size": 20,
        "total": 0,
        "total_pages": 0,
    }


def test_list_rejects_invalid_pagination_values() -> None:
    with prompt_client(FakePromptService()) as client:
        invalid_page = client.get("/prompts?page=0")
        invalid_page_size = client.get("/prompts?page_size=101")

    assert invalid_page.status_code == 422
    assert invalid_page_size.status_code == 422


def test_update_prompt_returns_the_updated_response() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        response = client.patch(
            f"/prompts/{prompt.id}",
            json={
                "title": "Welcome",
                "description": "A friendly prompt",
                "category": "marketing",
            },
        )

    assert response.status_code == 200
    assert response.json()["title"] == "Welcome"
    assert response.json()["description"] == "A friendly prompt"
    assert response.json()["category"] == "marketing"
    assert len(service.versions[prompt.id]) == 1


def test_prompt_version_history_returns_immutable_snapshots() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        client.patch(f"/prompts/{prompt.id}", json={"title": "Welcome"})
        client.patch(f"/prompts/{prompt.id}", json={"content": "Say hello warmly"})
        client.patch(f"/prompts/{prompt.id}", json={"content": "Say hello enthusiastically"})
        history_response = client.get(f"/prompts/{prompt.id}/versions")
        first_version_response = client.get(f"/prompts/{prompt.id}/versions/1")

    assert history_response.status_code == 200
    assert [item["version"] for item in history_response.json()] == [3, 2, 1]
    assert history_response.json()[1]["content"] == "Say hello warmly"
    assert first_version_response.status_code == 200
    assert first_version_response.json()["content"] == "Say hello"
    assert set(first_version_response.json()) == {
        "id",
        "prompt_id",
        "version",
        "content",
        "created_at",
    }


def test_prompt_version_routes_return_404_for_missing_prompts_or_versions() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        missing_prompt_response = client.get(f"/prompts/{uuid4()}/versions")
        missing_version_response = client.get(f"/prompts/{prompt.id}/versions/99")

    assert missing_prompt_response.status_code == 404
    assert missing_version_response.status_code == 404


def test_category_endpoint_returns_distinct_sorted_categories() -> None:
    service = FakePromptService()
    first = create_prompt(service)
    service.update_prompt(first.id, PromptUpdate(category="marketing"))
    service.create_prompt(PromptCreate(title="Second", content="Another prompt", category="general"))

    with prompt_client(service) as client:
        response = client.get("/prompts/categories")

    assert response.status_code == 200
    assert response.json() == ["general", "marketing"]


def test_favorite_routes_update_the_same_persisted_prompt() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        favorite_response = client.put(f"/prompts/{prompt.id}/favorite")
        unfavorite_response = client.delete(f"/prompts/{prompt.id}/favorite")

    assert favorite_response.status_code == 200
    assert favorite_response.json()["is_favorite"] is True
    assert unfavorite_response.status_code == 200
    assert unfavorite_response.json()["is_favorite"] is False
    assert service.get_prompt(prompt.id).is_favorite is False
    assert len(service.versions[prompt.id]) == 1


def test_template_routes_update_the_same_persisted_prompt() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        mark_response = client.put(f"/prompts/{prompt.id}/template")
        unmark_response = client.delete(f"/prompts/{prompt.id}/template")

    assert mark_response.status_code == 200
    assert mark_response.json()["is_template"] is True
    assert unmark_response.status_code == 200
    assert unmark_response.json()["is_template"] is False
    assert service.get_prompt(prompt.id).is_template is False
    assert len(service.versions[prompt.id]) == 1


def test_delete_prompt_returns_204_and_removes_the_prompt() -> None:
    service = FakePromptService()
    prompt = create_prompt(service)

    with prompt_client(service) as client:
        response = client.delete(f"/prompts/{prompt.id}")

    assert response.status_code == 204
    assert response.content == b""
    assert service.get_prompt(prompt.id) is None
    assert prompt.id not in service.versions


def test_prompt_routes_return_404_for_unknown_ids() -> None:
    missing_id = uuid4()

    with prompt_client(FakePromptService()) as client:
        get_response = client.get(f"/prompts/{missing_id}")
        patch_response = client.patch(f"/prompts/{missing_id}", json={"title": "Welcome"})
        delete_response = client.delete(f"/prompts/{missing_id}")
        favorite_response = client.put(f"/prompts/{missing_id}/favorite")
        template_response = client.put(f"/prompts/{missing_id}/template")

    assert get_response.status_code == 404
    assert patch_response.status_code == 404
    assert delete_response.status_code == 404
    assert favorite_response.status_code == 404
    assert template_response.status_code == 404
    assert get_response.json() == {"detail": "Prompt not found."}


def test_prompt_routes_use_pydantic_validation_for_invalid_requests() -> None:
    with prompt_client(FakePromptService()) as client:
        create_response = client.post("/prompts", json={"title": " ", "content": "Say hello"})
        update_response = client.patch(f"/prompts/{uuid4()}", json={})

    assert create_response.status_code == 422
    assert update_response.status_code == 422
