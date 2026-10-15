"""Pydantic v2 schemas for the officer case workflow API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChecklistItem(BaseModel):
    label: str
    done: bool = False


class OfficerCaseRead(BaseModel):
    """Persisted officer workflow state for one investigation."""

    model_config = ConfigDict(from_attributes=True)

    investigation_id: str
    review_decision: str
    review_notes: str | None = None
    reviewed_at: datetime | None = None
    response_status: str
    response_checklist: list[ChecklistItem] = Field(default_factory=list)
    report_status: str
    created_at: datetime
    updated_at: datetime | None = None


class OfficerCaseSummary(OfficerCaseRead):
    """A case plus the derived investigation facts the officer views need."""

    name: str
    status: str
    severity: str            # critical | high | moderate | low
    max_confidence: float
    positive_tiles: int
    area_km2: float
    detection_date: str | None = None
    has_source_reconstruction: bool = False
    has_attribution: bool = False
    alert_count: int = 0


class OfficerCaseUpdate(BaseModel):
    """Partial update — every field optional."""

    review_decision: str | None = None
    review_notes: str | None = None
    response_status: str | None = None
    response_checklist: list[ChecklistItem] | None = None
    report_status: str | None = None
