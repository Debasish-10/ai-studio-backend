from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Prompt(Base):
    """Current state of a persisted prompt-library entry."""

    __tablename__ = "prompts"
    __table_args__ = (
        CheckConstraint("length(trim(title)) > 0", name="ck_prompts_title_not_blank"),
        CheckConstraint("length(trim(content)) > 0", name="ck_prompts_content_not_blank"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="general",
        server_default="general",
        index=True,
    )
    is_favorite: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    is_template: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    versions: Mapped[list["PromptVersion"]] = relationship(
        back_populates="prompt",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="PromptVersion.version",
    )
