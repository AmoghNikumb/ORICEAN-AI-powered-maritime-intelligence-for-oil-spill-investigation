"""
Pydantic v2 schemas for the Source Reconstruction pipeline.

Two operational schemas are provided:

- ``SourceReconstructionRequest``  — payload for POST /api/source/reconstruct
- ``SourceReconstructionResponse`` — full reconstruction result returned from the API

These models are used by ``backend/app/api/source.py`` and produced by
``backend/app/services/drift_service.py``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SourceReconstructionRequest(BaseModel):
    """Request payload for triggering kinematic drift backtracking on an investigation."""

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation whose spill detections will be backtracked.",
    )
    drift_speed_m_s: float = Field(
        default=0.3,
        ge=0.01,
        le=5.0,
        description=(
            "Drift speed in m/s for the kinematic model. "
            "Overrides the DRIFT_SPEED_M_S setting when provided. "
            "Valid range: [0.01, 5.0]."
        ),
    )
    lookback_hours: int = Field(
        default=48,
        ge=1,
        le=336,
        description=(
            "Number of hours to backtrack from the satellite acquisition time. "
            "Overrides the DRIFT_LOOKBACK_HOURS setting when provided. "
            "Valid range: [1, 336] (up to 14 days)."
        ),
    )
    force_rerun: bool = Field(
        default=False,
        description=(
            "When True, recomputes and overwrites any existing "
            "source_reconstruction_result for this investigation."
        ),
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SourceReconstructionResponse(BaseModel):
    """Full source reconstruction result returned by POST /api/source/reconstruct."""

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation this reconstruction belongs to.",
    )
    source_corridor_geojson: dict = Field(
        ...,
        description=(
            "GeoJSON Polygon or MultiPolygon representing the probable source corridor, "
            "derived by buffering the spill detection region by the computed drift distance."
        ),
    )
    release_window_start: str = Field(
        ...,
        description="Earliest probable release time (ISO-8601 UTC). Equals T₀ − lookback_hours.",
    )
    release_window_end: str = Field(
        ...,
        description=(
            "Latest probable release time (ISO-8601 UTC). "
            "Equals T₀ − (lookback_hours × 0.5)."
        ),
    )
    source_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the source corridor estimate in [0.0, 1.0]. "
            "Derived from the ratio of drift uncertainty to drift distance."
        ),
    )
    reconstruction_method: str = Field(
        ...,
        description=(
            "Identifier for the reconstruction algorithm used. "
            "Always 'kinematic_drift_mvp' for the current implementation."
        ),
    )
    drift_distance_km: float = Field(
        ...,
        ge=0.0,
        description="Total drift distance in km computed from speed × lookback time (capped at 500 km).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "List of advisory warnings about this reconstruction run, e.g. "
            "'no_ocean_current_data_used'."
        ),
    )
    run_at: datetime = Field(
        ...,
        description="UTC timestamp of when this reconstruction was computed.",
    )
