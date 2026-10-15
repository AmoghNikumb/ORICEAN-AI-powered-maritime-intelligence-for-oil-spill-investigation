"""Pydantic v2 schemas for the officer alerting API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AlertRead(BaseModel):
    """A recorded alert and its Twilio delivery outcome."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    investigation_id: str | None = None
    created_at: datetime
    severity: str
    headline: str
    region_label: str | None = None
    detection_time: str | None = None
    confidence: float | None = None
    positive_tiles: int | None = None
    area_km2: float | None = None
    investigation_url: str | None = None
    delivery_status: str
    delivery_detail: str | None = None
    recipients_count: int = 0


class NotificationSettingsRead(BaseModel):
    """Current alert trigger configuration and recipient list."""

    model_config = ConfigDict(from_attributes=True)

    enabled: bool
    min_confidence: float
    min_positive_tiles: int
    recipients: list[str]
    updated_at: datetime | None = None


class NotificationSettingsUpdate(BaseModel):
    """Partial update for the notification settings (all fields optional)."""

    enabled: bool | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_positive_tiles: int | None = Field(default=None, ge=0)
    recipients: list[str] | None = None


class AlertStatus(BaseModel):
    """Twilio readiness — safe to expose to the officer UI (no secrets)."""

    package_installed: bool
    account_configured: bool
    from_number_set: bool
    recipients_count: int
    alerting_enabled: bool
