from collections.abc import Mapping
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import Prompt


class PromptRepository:
    """Database access for persisted prompt-library entries."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, prompt: Prompt) -> Prompt:
        self.session.add(prompt)
        self.session.flush()
        return prompt

    def get_by_id(self, prompt_id: UUID) -> Prompt | None:
        return self.session.get(Prompt, prompt_id)

    def list_prompts(
        self,
        *,
        search: str | None,
        category: str | None,
        is_favorite: bool | None,
        is_template: bool | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Prompt], int]:
        filters = self._list_filters(
            search=search,
            category=category,
            is_favorite=is_favorite,
            is_template=is_template,
        )
        statement = (
            select(Prompt)
            .where(*filters)
            .order_by(Prompt.created_at.desc(), Prompt.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        total_statement = select(func.count()).select_from(Prompt).where(*filters)

        return list(self.session.scalars(statement).all()), self.session.scalar(total_statement) or 0

    def list_categories(self) -> list[str]:
        statement = select(Prompt.category).distinct().order_by(Prompt.category)
        return list(self.session.scalars(statement).all())

    def update(self, prompt: Prompt, values: Mapping[str, object]) -> Prompt:
        for field, value in values.items():
            setattr(prompt, field, value)
        self.session.flush()
        return prompt

    def delete(self, prompt: Prompt) -> None:
        self.session.delete(prompt)
        self.session.flush()

    @staticmethod
    def _list_filters(
        *,
        search: str | None,
        category: str | None,
        is_favorite: bool | None,
        is_template: bool | None,
    ) -> list[object]:
        filters = []
        normalized_search = search.strip() if search else None
        if normalized_search:
            pattern = f"%{normalized_search}%"
            filters.append(
                or_(
                    Prompt.title.ilike(pattern),
                    Prompt.content.ilike(pattern),
                    Prompt.description.ilike(pattern),
                )
            )
        if category:
            filters.append(Prompt.category == category)
        if is_favorite is not None:
            filters.append(Prompt.is_favorite == is_favorite)
        if is_template is not None:
            filters.append(Prompt.is_template == is_template)
        return filters
