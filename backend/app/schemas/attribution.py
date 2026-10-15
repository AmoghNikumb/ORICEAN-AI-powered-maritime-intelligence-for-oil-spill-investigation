"""
Pydantic v2 schemas for the Vessel Attribution pipeline.

Models are ordered by dependency (leaf → root):

- ``VesselRecord``       — internal AIS provider vessel record with track points
- ``AISQueryResult``     — result returned from the AIS service layer
- ``ScoreBreakdown``     — four sub-scores and their weights for a single candidate
- ``RankedCandidate``    — fully scored and ranked attribution candidate
- ``AttributionRequest`` — payload for POST /api/attribution/run
- ``AttributionResponse``— response from POST /api/attribution/run

These models are used by ``backend/app/api/attribution.py`` and produced by
``backend/app/services/attribution_service.py`` and ``backend/app/services/ais_service.py``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# AIS data models (internal — not exposed directly in API responses)
# ---------------------------------------------------------------------------


class VesselRecord(BaseModel):
    """
    Internal model representing a vessel and its AIS track returned by the
    AIS provider adapter.

    ``track`` is a list of (longitude, latitude, timestamp_utc) tuples where
    ``timestamp_utc`` is an ISO-8601 UTC string.
    """

    mmsi: str = Field(..., description="Maritime Mobile Service Identity — 9-digit unique vessel identifier.")
    name: str | None = Field(default=None, description="Vessel name as reported in AIS transmissions.")
    imo: str | None = Field(default=None, description="IMO vessel registration number, if available.")
    vessel_type: str | None = Field(default=None, description="AIS vessel type category (e.g. 'Tanker', 'Cargo').")
    flag: str | None = Field(default=None, description="ISO 3166-1 alpha-2 flag state code (e.g. 'PA', 'LR').")
    track: list[tuple[float, float, str]] = Field(
        default_factory=list,
        description=(
            "Ordered list of track points as (longitude, latitude, timestamp_utc). "
            "``timestamp_utc`` is an ISO-8601 UTC string."
        ),
    )
    activity_hours: float | None = Field(
        default=None,
        description=(
            "Provider-reported hours of AIS activity for this vessel inside the "
            "search area during the query window (e.g. GFW 4wings 'hours'). "
            "``None`` when the provider does not report a dwell figure."
        ),
    )
    position_count: int | None = Field(
        default=None,
        description="Number of AIS position/activity records the provider returned for this vessel.",
    )


class AISQueryResult(BaseModel):
    """
    Aggregate result from a single AIS provider query.

    ``warnings`` carries non-fatal diagnostic strings such as
    ``'no_ais_provider_configured'`` or ``'unsupported_geometry_type'``.
    """

    vessels: list[VesselRecord] = Field(
        default_factory=list,
        description="All vessel records returned (or synthesised) for the queried region and time window.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during the AIS query (e.g. missing credentials, unknown provider).",
    )


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """
    Transparent breakdown of the four sub-scores that contribute to an
    ``attribution_score``.

    Sub-score ranges:
    - ``spatial_score``            : [0.0, 1.0] — proximity to the source region
    - ``temporal_score``          : [0.0, 1.0] — fraction of the lookback window covered
    - ``trajectory_score``        : [0.0, 1.0] — track overlap / heading alignment with the corridor
    - ``time_near_region_score``  : [0.0, 1.0] — dwell time inside the search area vs the lookback window

    ``weights`` maps each dimension name to its contribution weight; values
    must sum to 1.0 ±0.001.
    """

    spatial_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Spatial proximity score in [0.0, 1.0]; higher when closer to the source region.",
    )
    temporal_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Temporal overlap score in [0.0, 1.0]; fraction of the lookback window during which the vessel was present.",
    )
    trajectory_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Trajectory-overlap score in [0.0, 1.0]; fraction of the vessel's positions that fall inside the search buffer (falls back to heading alignment with the drift vector).",
    )
    time_near_region_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dwell score in [0.0, 1.0]; hours of AIS activity inside the search area divided by the lookback-window length, capped at 1.0.",
    )
    weights: dict[str, float] = Field(
        ...,
        description=(
            "Weight applied to each sub-score when computing attribution_score, "
            "e.g. {'spatial': 0.35, 'temporal': 0.30, 'trajectory': 0.25, 'time_near_region': 0.10}. "
            "Values must sum to 1.0 ±0.001."
        ),
    )


class RankedCandidate(BaseModel):
    """
    A fully scored and ranked attribution candidate vessel.

    ``attribution_score`` is in [0.0, 100.0].
    ``evidence`` contains at least four human-readable strings (one per
    sub-score dimension) explaining why the candidate received its score.
    """

    rank: int = Field(..., ge=1, description="1-based rank position; rank 1 is the highest-scoring candidate.")
    mmsi: str = Field(..., description="Maritime Mobile Service Identity — 9-digit unique vessel identifier.")
    name: str | None = Field(default=None, description="Vessel name as reported in AIS transmissions.")
    imo: str | None = Field(default=None, description="IMO vessel registration number, if available.")
    vessel_type: str | None = Field(default=None, description="AIS vessel type category (e.g. 'Tanker', 'Cargo').")
    flag: str | None = Field(default=None, description="ISO 3166-1 alpha-2 flag state code (e.g. 'PA', 'LR').")
    attribution_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite attribution score in [0.0, 100.0]; higher score = higher suspicion.",
    )
    min_distance_km: float = Field(
        ...,
        ge=0.0,
        description="Minimum distance in kilometres between any track point and the source region boundary.",
    )
    entry_time: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp of the first track point found inside the source corridor.",
    )
    exit_time: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp of the last track point found inside the source corridor.",
    )
    closest_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp of the AIS position that came closest to the source region.",
    )
    time_diff_hours: float | None = Field(
        default=None,
        description="Absolute hours between the satellite detection time and ``closest_timestamp``.",
    )
    num_ais_positions: int = Field(
        default=0,
        ge=0,
        description="Number of AIS position/activity records the provider returned for this vessel.",
    )
    trajectory_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the vessel's AIS positions that fall inside the search buffer around the source region.",
    )
    time_near_region_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate hours the vessel spent within the search area during the lookback window.",
    )
    points_near_source: int = Field(
        default=0,
        ge=0,
        description="Number of in-window AIS positions within the search buffer of the source corridor.",
    )
    qualified: bool = Field(
        default=True,
        description=(
            "True when the candidate cleared every evidence gate (closest approach, "
            "time from detection, trajectory overlap, AIS points near the source, "
            "minimum score). Qualified candidates are shown by default; the rest are "
            "available under 'show all'."
        ),
    )
    qualification_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable statement of each evidence gate this candidate passed.",
    )
    track: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Ordered (longitude, latitude) positions for drawing the vessel's historical AIS track on the map.",
    )
    track_points: list[tuple[float, float, str]] = Field(
        default_factory=list,
        description=(
            "Ordered (longitude, latitude, timestamp_utc) points for the vessel's "
            "real historical AIS trajectory — used by the ship-route time-slider replay. "
            "``timestamp_utc`` is an ISO-8601 UTC string."
        ),
    )
    score_breakdown: ScoreBreakdown = Field(
        ...,
        description="Detailed sub-score breakdown with individual scores and their weights.",
    )
    evidence: list[str] = Field(
        ...,
        min_length=4,
        description=(
            "Human-readable evidence strings explaining each sub-score dimension "
            "(at least four items — one per dimension)."
        ),
    )


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class AttributionRequest(BaseModel):
    """Request payload for POST /api/attribution/run."""

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation for which vessel attribution should be run.",
    )
    force_rerun: bool = Field(
        default=False,
        description="When True, re-runs attribution even if a previous result is already stored.",
    )


class AttributionResponse(BaseModel):
    """
    Complete output of a vessel attribution run.

    Returned by POST /api/attribution/run and GET /api/attribution/{id}.
    Also persisted to ``Investigation.attribution_result`` as a JSONB document.
    """

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation this attribution result belongs to.",
    )
    candidates: list[RankedCandidate] = Field(
        default_factory=list,
        description=(
            "Ranked list of *potential source vessel candidates* — vessels whose historical "
            "AIS activity is spatially and temporally consistent with the detected spill. "
            "Empty when AIS is unconfigured, the provider/API failed, or no vessels matched. "
            "A ranking is NOT a determination that a vessel caused the spill."
        ),
    )
    total_candidates: int = Field(
        ...,
        ge=0,
        description="Total number of vessels that were scored (passed the initial spatial + temporal filter).",
    )
    qualified_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of scored vessels that cleared every evidence gate. These are "
            "the candidates surfaced by default in the UI (top N by score)."
        ),
    )
    ais_provider_configured: bool = Field(
        ...,
        description="True when a usable AIS provider is configured (e.g. AIS_PROVIDER=gfw with GFW_API_TOKEN set).",
    )
    data_source: str = Field(
        default="none",
        description="Which historical AIS source produced this result: 'gfw', 'marinecadastre', 'aisstream', or 'none'.",
    )
    search_region_geojson: dict | None = Field(
        default=None,
        description="GeoJSON geometry of the spill/source region used for the vessel search.",
    )
    release_window_start: str | None = Field(
        default=None,
        description="ISO-8601 UTC start of the historical AIS search window (detection time minus the lookback).",
    )
    release_window_end: str | None = Field(
        default=None,
        description="ISO-8601 UTC end of the historical AIS search window (the satellite detection time).",
    )
    lookback_hours: int | None = Field(
        default=None,
        description="Lookback window length in hours used for the historical AIS search.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings from the AIS query or attribution pipeline.",
    )
    run_at: datetime = Field(
        ...,
        description="UTC datetime when this attribution run was executed.",
    )
