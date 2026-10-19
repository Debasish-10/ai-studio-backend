from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import PromptVersion


class PromptVersionRepository:
    """Database access for immutable PromptVersion snapshots."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, prompt_version: PromptVersion) -> PromptVersion:
        self.session.add(prompt_version)
        self.session.flush()
        return prompt_version

    def next_version_number(self, prompt_id: UUID) -> int:
        latest = self.session.scalar(
            select(func.max(PromptVersion.version)).where(PromptVersion.prompt_id == prompt_id)
        )
        return (latest or 0) + 1

    def list_for_prompt(self, prompt_id: UUID) -> list[PromptVersion]:
        statement = (
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version.desc())
        )
        return list(self.session.scalars(statement).all())

    def get_by_prompt_and_version(self, prompt_id: UUID, version: int) -> PromptVersion | None:
        statement = select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == version,
        )
        return self.session.scalar(statement)
