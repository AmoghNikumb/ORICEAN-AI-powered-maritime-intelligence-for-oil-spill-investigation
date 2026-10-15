"""
Drift service for ORICEAN — kinematic oil-spill source reconstruction.

Responsibilities:
- Derive a seed geometry from the ML-produced spill detections stored on an
  Investigation record.
- Apply a simple uniform-drift backtracking model to estimate the probable
  geographic release corridor (Source_Corridor) as a GeoJSON Polygon or
  MultiPolygon.
- Compute a Release_Window ``[T₀ − lookback_hours, T₀ − lookback_hours×0.5]``
  and a ``source_confidence`` float derived from the drift uncertainty.

Exported symbols:
- ``DriftServiceError``         — custom exception (HTTP 422 by default)
- ``SourceReconstructionOutput``— dataclass holding the reconstruction result
- ``compute_source_corridor()`` — async end-to-end reconstruction orchestrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# 1 degree of latitude/longitude ≈ 111 km at the equator — used for degree↔km conversion.
_DEG_TO_KM: float = 111.0

# Maximum corridor expansion radius to avoid absurdly large polygons.
_MAX_DRIFT_KM: float = 500.0

# Fixed variance fraction for the kinematic uncertainty model (±50% speed uncertainty).
_SPEED_VARIANCE_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DriftServiceError(Exception):
    """Raised when the Drift_Service cannot fulfil a reconstruction request.

    Attributes
    ----------
    message:
        Human-readable description of the error.
    status_code:
        Suggested HTTP status code (default 422 — Unprocessable Entity).
    """

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceReconstructionOutput:
    """Container for the result of a kinematic drift backtracking computation.

    Attributes
    ----------
    source_corridor_geojson:
        GeoJSON dict (Polygon or MultiPolygon) representing the probable
        geographic region from which the spill originated.
    release_window_start:
        Earliest estimated discharge time as a timezone-aware UTC datetime.
    release_window_end:
        Latest estimated discharge time as a timezone-aware UTC datetime.
    source_confidence:
        Confidence score in ``[0.0, 1.0]`` derived from the drift uncertainty.
    reconstruction_method:
        Label identifying the algorithm; always ``"kinematic_drift_mvp"`` for
        this implementation.
    drift_distance_km:
        Effective corridor expansion radius in kilometres (capped at 500 km).
    warnings:
        List of advisory strings included in the API response.
    """

    source_corridor_geojson: dict
    release_window_start: datetime
    release_window_end: datetime
    source_confidence: float
    reconstruction_method: str = "kinematic_drift_mvp"
    drift_distance_km: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_seed_geometry(
    spill_detections: list[dict],
) -> tuple:
    """Derive the seed Shapely geometry and primary centroid from ML detections.

    For a single detection the polygon is used directly.  For two or more
    detections the union of all polygons is computed as the seed, and the
    centroid of the detection with the highest ``mean_confidence`` (tiebreak:
    largest ``area_km2``; fallback when all confidences are null: largest
    ``area_km2``) is used as the primary centroid.

    Parameters
    ----------
    spill_detections:
        Raw list of detection dicts from ``investigation.ml_result["spill_detections"]``.
        Each dict is expected to contain a ``"polygon_geojson"`` key with a GeoJSON
        geometry, and optionally ``"mean_confidence"`` (float | None) and
        ``"area_km2"`` (float | None).

    Returns
    -------
    tuple[shapely.geometry.base.BaseGeometry, shapely.geometry.Point]
        ``(seed_geometry, primary_centroid)``

    Raises
    ------
    DriftServiceError
        If ``spill_detections`` is empty (HTTP 422).
    """
    if not spill_detections:
        raise DriftServiceError(
            "No spill detections found in investigation — run ML inference first",
            status_code=422,
        )

    if len(spill_detections) == 1:
        seed_geom = shape(spill_detections[0]["polygon_geojson"])
        return seed_geom, seed_geom.centroid

    # Two or more detections — build union seed.
    shapely_geoms = [shape(d["polygon_geojson"]) for d in spill_detections]
    seed_geom = unary_union(shapely_geoms)

    # Select primary centroid: highest mean_confidence, tiebreak by area_km2.
    all_null_confidence = all(
        d.get("mean_confidence") is None for d in spill_detections
    )

    if all_null_confidence:
        # Fallback: largest area_km2.
        winner = max(
            spill_detections,
            key=lambda d: d.get("area_km2") or 0.0,
        )
    else:
        winner = max(
            spill_detections,
            key=lambda d: (
                d.get("mean_confidence") if d.get("mean_confidence") is not None else -1.0,
                d.get("area_km2") or 0.0,
            ),
        )

    primary_centroid: Point = shape(winner["polygon_geojson"]).centroid
    return seed_geom, primary_centroid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def _fetch_wind_drift_speed(
    lon: float,
    lat: float,
) -> tuple[float, str]:
    """Fetch live wind speed from Open-Meteo and derive surface drift speed.

    Uses the Open-Meteo free weather API — no key required.
    Returns (drift_speed_m_s, warning_string).

    The drift speed is approximated using the standard 3% wind-to-surface-drift
    ratio used in oil spill trajectory modelling:
        drift_speed ≈ wind_speed_10m × 0.033

    If the fetch fails for any reason, falls back to the default 0.3 m/s
    and returns an appropriate warning.

    Args:
        lon: Longitude of the spill centroid.
        lat: Latitude of the spill centroid.

    Returns:
        Tuple of (drift_speed_m_s, source_description_string).
    """
    if not _HTTPX_AVAILABLE:
        return 0.3, "no_ocean_current_data_used"

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=wind_speed_10m"
        f"&wind_speed_unit=ms"
        f"&forecast_days=1"
        f"&timezone=UTC"
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            speeds = data.get("hourly", {}).get("wind_speed_10m", [])
            # Use the first available (current) hour's wind speed
            valid_speeds = [s for s in speeds if s is not None and s >= 0]
            if not valid_speeds:
                return 0.3, "no_ocean_current_data_used"
            wind_speed_m_s = valid_speeds[0]
            # 3.3% wind-to-drift ratio (NOAA standard for oil spill response)
            drift_speed = max(0.01, min(5.0, wind_speed_m_s * 0.033))
            logger.info(
                "Open-Meteo wind at (%.4f, %.4f): %.2f m/s → drift %.4f m/s",
                lon, lat, wind_speed_m_s, drift_speed,
            )
            return drift_speed, f"wind_drift_from_open_meteo_wind_{wind_speed_m_s:.1f}ms"
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed (%s); using default drift speed.", exc)
        return 0.3, "no_ocean_current_data_used"


async def compute_source_corridor(
    spill_detections: list[dict],
    acquisition_timestamp: datetime,
    drift_speed_m_s: float = 0.3,
    lookback_hours: int = 48,
) -> SourceReconstructionOutput:
    """Compute the probable source corridor for an oil spill via kinematic drift backtracking.

    The algorithm applies a uniform backward drift offset to the detected spill
    geometry to produce a Source_Corridor GeoJSON polygon.  No real ocean current
    data is used in this MVP; a ``"no_ocean_current_data_used"`` advisory is always
    included in the response warnings.

    Parameters
    ----------
    spill_detections:
        Raw detection dicts from ``investigation.ml_result["spill_detections"]``.
        Must contain at least one entry; each entry must have a valid
        ``"polygon_geojson"`` key.
    acquisition_timestamp:
        The SAR acquisition time used as reference time ``T₀``.  Should be
        timezone-aware (UTC); if naive it is treated as UTC.
    drift_speed_m_s:
        Assumed drift speed in metres per second.  Must be in ``[0.01, 5.0]``
        (default 0.3 m/s).
    lookback_hours:
        Number of hours to backtrack from ``T₀``.  Must be in ``[1, 336]``
        (default 48 h).

    Returns
    -------
    SourceReconstructionOutput

    Raises
    ------
    DriftServiceError
        If parameter validation fails (HTTP 422) or the detection list is empty.
    """
    # --- Parameter validation -----------------------------------------------
    if not (0.01 <= drift_speed_m_s <= 5.0) or not (1 <= lookback_hours <= 336):
        raise DriftServiceError(
            "Invalid drift parameters: drift_speed_m_s must be in [0.01, 5.0] "
            "and lookback_hours must be in [1, 336]",
            status_code=422,
        )

    # --- Fetch live wind drift speed from Open-Meteo (when using default) ------
    weather_warning: str = "no_ocean_current_data_used"
    _used_live_weather = False
    if abs(drift_speed_m_s - 0.3) < 1e-9:
        # Only fetch live data when the caller is using the default speed,
        # so explicit overrides (from env config) are always respected.
        # We need a centroid to query — derive it from spill_detections first.
        try:
            _seed_geom_temp, _centroid_temp = _derive_seed_geometry(spill_detections)
            fetched_speed, weather_warning = await _fetch_wind_drift_speed(
                _centroid_temp.x, _centroid_temp.y
            )
            if "open_meteo" in weather_warning:
                drift_speed_m_s = fetched_speed
                _used_live_weather = True
        except Exception as _exc:
            logger.warning("Could not fetch live drift speed: %s", _exc)

    # --- Derive seed geometry -------------------------------------------------
    seed_geom, _centroid = _derive_seed_geometry(spill_detections)

    # --- Kinematic drift computation -----------------------------------------
    lookback_seconds: float = lookback_hours * 3600.0
    drift_distance_km: float = min(
        drift_speed_m_s * lookback_seconds / 1000.0,
        _MAX_DRIFT_KM,
    )

    # Expand seed geometry in all directions to produce the source corridor.
    corridor_geom = seed_geom.buffer(drift_distance_km / _DEG_TO_KM)

    # --- Release window -------------------------------------------------------
    # Ensure acquisition_timestamp is timezone-aware (treat naive as UTC).
    if acquisition_timestamp.tzinfo is None:
        acquisition_timestamp = acquisition_timestamp.replace(tzinfo=timezone.utc)

    release_window_start: datetime = acquisition_timestamp - timedelta(hours=lookback_hours)
    release_window_end: datetime = acquisition_timestamp - timedelta(
        hours=lookback_hours * 0.5
    )

    # --- Source confidence ----------------------------------------------------
    drift_uncertainty_km: float = _SPEED_VARIANCE_FRACTION * drift_speed_m_s * lookback_seconds / 1000.0
    # Simplifies to 0.5 for any nonzero drift_distance_km; written in full
    # so future parametric changes (e.g. variable variance fraction) work correctly.
    raw_confidence: float = 1.0 - (drift_uncertainty_km / drift_distance_km)
    source_confidence: float = max(0.0, min(1.0, raw_confidence))

    # --- Build output ---------------------------------------------------------
    warnings: list[str] = [weather_warning]
    if _used_live_weather:
        warnings.append("drift_speed_derived_from_live_wind_data")
    else:
        pass  # weather_warning already describes the fallback

    logger.info(
        "Source corridor computed: drift_distance_km=%.2f, confidence=%.3f, "
        "release_window=[%s, %s]",
        drift_distance_km,
        source_confidence,
        release_window_start.isoformat(),
        release_window_end.isoformat(),
    )

    return SourceReconstructionOutput(
        source_corridor_geojson=dict(mapping(corridor_geom)),
        release_window_start=release_window_start,
        release_window_end=release_window_end,
        source_confidence=source_confidence,
        reconstruction_method="kinematic_drift_mvp",
        drift_distance_km=drift_distance_km,
        warnings=warnings,
    )
