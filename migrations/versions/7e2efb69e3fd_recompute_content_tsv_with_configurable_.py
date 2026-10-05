"""recompute content_tsv with configurable ts_config

Revision ID: 7e2efb69e3fd
Revises: 14ece462915a
Create Date: 2026-10-05 22:36:01.069783

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7e2efb69e3fd"
down_revision: str | None = "14ece462915a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE chunks "
            "SET content_tsv = to_tsvector('english', content) "
            "WHERE content IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE chunks "
            "SET content_tsv = to_tsvector('simple', content) "
            "WHERE content IS NOT NULL"
        )
    )
