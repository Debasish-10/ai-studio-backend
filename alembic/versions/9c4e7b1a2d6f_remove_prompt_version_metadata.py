"""remove prompt version metadata

Revision ID: 9c4e7b1a2d6f
Revises: a9f509f6feed
Create Date: 2026-10-08 20:15:00.000000
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c4e7b1a2d6f"
down_revision: str | Sequence[str] | None = "a9f509f6feed"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Keep only historical prompt content in version snapshots."""
    op.drop_column("prompt_versions", "is_template")
    op.drop_column("prompt_versions", "is_favorite")
    op.drop_column("prompt_versions", "category")
    op.drop_column("prompt_versions", "description")
    op.drop_column("prompt_versions", "title")


def downgrade() -> None:
    """Restore the former columns using the current parent Prompt metadata."""
    op.add_column(
        "prompt_versions",
        sa.Column("title", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "prompt_versions",
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.add_column(
        "prompt_versions",
        sa.Column("category", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "prompt_versions",
        sa.Column(
            "is_favorite",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "prompt_versions",
        sa.Column(
            "is_template",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE prompt_versions AS prompt_version
            SET
                title = prompt.title,
                description = prompt.description,
                category = prompt.category,
                is_favorite = prompt.is_favorite,
                is_template = prompt.is_template
            FROM prompts AS prompt
            WHERE prompt_version.prompt_id = prompt.id
            """
        )
    )
    op.alter_column("prompt_versions", "title", existing_type=sa.String(length=255), nullable=False)
    op.alter_column("prompt_versions", "category", existing_type=sa.String(length=100), nullable=False)
