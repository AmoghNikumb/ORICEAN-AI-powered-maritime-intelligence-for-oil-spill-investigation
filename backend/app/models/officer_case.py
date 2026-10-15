"""
SQLAlchemy ORM model for the officer case workflow.

One ``OfficerCase`` row per investigation an officer is working — it carries the
officer's review decision, the field-response state, and the report-handoff
state. Rows are created lazily on first access. The table is created by
``Base.metadata.create_all`` at startup (same as ``alerts``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

# Default containment checklist seeded into every new case row.
DEFAULT_CHECKLIST: list[dict] = [
    {"label": "Confirm slick perimeter from Sentinel-1", "done": False},
    {"label": "Notify regional response centre", "done": False},
    {"label": "Task verification vessel / sampling", "done": False},
    {"label": "Deploy containment boom", "done": False},
    {"label": "Schedule next satellite pass", "done": False},
]

REVIEW_DECISIONS = ("pending", "approved", "returned", "escalated")
RESPONSE_STATUSES = ("not_started", "responding", "contained", "closed")
REPORT_STATUSES = ("draft", "awaiting_review", "approved", "distributed")


class OfficerCase(Base):
    """Officer-side workflow state for a single investigation."""

    __tablename__ = "officer_cases"

    investigation_id: Mapped[str] = mapped_column(String, primary_key=True)

    review_decision: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    review_notes: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    response_status: Mapped[str] = mapped_column(String, nullable=False, default="not_started")
    response_checklist: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    report_status: Mapped[str] = mapped_column(String, nullable=False, default="draft")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<OfficerCase inv={self.investigation_id!r} review={self.review_decision!r} "
            f"response={self.response_status!r} report={self.report_status!r}>"
        )
