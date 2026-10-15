"""
Pydantic v2 schemas for the Investigation resource.

Three operational schemas are provided:
- ``InvestigationCreate``  — payload for POST /investigations
- ``InvestigationUpdate``  — payload for PATCH /investigations/{id}
- ``InvestigationRead``    — full response model (ORM-compatible)

Plus the pagination wrapper:
- ``PaginatedInvestigations`` — list response with metadata
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class InvestigationCreate(BaseModel):
    """Fields required to create a new Investigation."""

    name: str
    aoi_geojson: dict
    start_date: str
    end_date: str


# ---------------------------------------------------------------------------
# Update (all optional — PATCH semantics)
# ---------------------------------------------------------------------------


class InvestigationUpdate(BaseModel):
    """
    Partial update payload.

    Only fields explicitly included in the request body will be applied;
    all others retain their current database values.
    """

    name: str | None = None
    aoi_geojson: dict | None = None
    start_date: str | None = None
    end_date: str | None = None
    status: str | None = None
    ml_result: dict | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Read (full record)
# ---------------------------------------------------------------------------


class InvestigationRead(BaseModel):
    """Full Investigation representation returned from the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    aoi_geojson: dict
    start_date: str
    end_date: str
    status: str
    ml_result: dict | None
    ml_run_at: datetime | None
    error_message: str | None
    source_reconstruction_result: dict | None = None
    attribution_result: dict | None = None
    created_at: datetime | None
    updated_at: datetime | None


# ---------------------------------------------------------------------------
# Pagination wrapper
# ---------------------------------------------------------------------------


class PaginatedInvestigations(BaseModel):
    """Paginated list of investigations with cursor metadata."""

    items: list[InvestigationRead]
    total: int
    page: int
    page_size: int
