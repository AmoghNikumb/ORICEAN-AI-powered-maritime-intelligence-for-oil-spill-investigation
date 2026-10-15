"""
SQLAlchemy ORM models for officer alerting.

- ``Alert``               — one row per significant-detection (or test) alert,
                            with its Twilio delivery outcome.
- ``NotificationSettings`` — single-row (id=1) runtime config for the alert
                            trigger and recipient list. Seeded from environment
                            defaults on first read.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Alert(Base):
    """A recorded officer alert for a significant oil-spill detection."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    investigation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # "test" for manual test sends, otherwise a severity bucket.
    severity: Mapped[str] = mapped_column(String, nullable=False, default="moderate")

    headline: Mapped[str] = mapped_column(String, nullable=False)
    region_label: Mapped[str | None] = mapped_column(String, nullable=True)
    detection_time: Mapped[str | None] = mapped_column(String, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    positive_tiles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    area_km2: Mapped[float | None] = mapped_column(Float, nullable=True)

    investigation_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # sent | partial | failed | skipped | not_configured
    delivery_status: Mapped[str] = mapped_column(String, nullable=False, default="skipped")
    delivery_detail: Mapped[str | None] = mapped_column(String, nullable=True)
    recipients_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Alert id={self.id!r} severity={self.severity!r} status={self.delivery_status!r}>"


class NotificationSettings(Base):
    """Single-row runtime configuration for officer alerting (``id == 1``)."""

    __tablename__ = "notification_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    min_positive_tiles: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    # Comma-separated E.164 phone numbers.
    recipients: Mapped[str] = mapped_column(String, nullable=False, default="")

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NotificationSettings enabled={self.enabled} "
            f"min_confidence={self.min_confidence} recipients={self.recipients!r}>"
        )
