"""
Pydantic v2 schema for the sanitised public incident feed.

Deliberately excludes everything operational: AOI internals, tile results,
Grad-CAM, source reconstruction, all vessel-attribution data, numeric model
scores, warnings, and the raw ``ml_result``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PublicIncident(BaseModel):
    """A single public-facing spill incident."""

    id: str
    region_label: str
    status: str
    detection_date: str | None = None
    centroid: list[float] | None = None            # [lon, lat]
    spill_region_geojson: dict | None = None       # merged detection polygon only
    severity: str                                  # critical | high | moderate | low
    confidence_band: str                           # high | moderate | low
    area_km2: float | None = None
    updated_at: datetime | None = None
