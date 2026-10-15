"""
FastAPI router for the Vessel Attribution pipeline.

Routes are intentionally thin — all business logic lives in
``attribution_service`` and ``ais_service``.
Every successful response is wrapped in ``{"data": ...}`` and every error
is wrapped in ``{"error": ..., "detail": ...}``.

Routes
------
POST /api/attribution/run   — run AIS vessel attribution for an investigation
GET  /api/attribution/{investigation_id} — retrieve a stored attribution result
"""

from __future__ import annotations

import datetime
import logging
import traceback
from datetime import timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from shapely.geometry import shape
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import attribution_weights, settings
from app.db.database import get_db
from app.schemas.attribution import AttributionRequest, AttributionResponse
from app.services import ais_service, attribution_service, investigation_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/attribution", tags=["attribution"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a consistently formatted JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail},
    )


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_attribution(
    body: AttributionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Run AIS vessel attribution for an investigation.

    Steps
    -----
    1. Look up the investigation; 404 if missing.
    2. Return 409 if attribution already exists and ``force_rerun`` is False.
    3. Determine search region from source_reconstruction_result or ml_result; 422 if neither.
    4. Determine release window from source_reconstruction_result or ml_result timestamps.
    5. Query AIS vessels via ``ais_service.query_vessels()``.
    6. Filter candidates via ``attribution_service.filter_candidates()``.
    7. Score and rank candidates via ``attribution_service.score_candidates()``.
    8. Build ``AttributionResponse``, persist to investigation, commit, and return.
    9. On any unexpected exception, return 500.
    """
    # --- Step 1: Validate investigation exists --------------------------------
    investigation = await investigation_service.get_investigation(
        db, body.investigation_id
    )
    if investigation is None:
        return _error_response(404, "Not Found", "Investigation not found.")

    # --- Step 2: Guard against re-runs without force_rerun --------------------
    if investigation.attribution_result and not body.force_rerun:
        return _error_response(
            409,
            "Conflict",
            "Attribution already completed. Pass force_rerun=true to re-run.",
        )

    # --- Step 3: Determine search region --------------------------------------
    source_recon = investigation.source_reconstruction_result
    ml_result = investigation.ml_result

    search_region: dict | None = None

    if source_recon is not None:
        search_region = source_recon.get("source_corridor_geojson")

    if search_region is None and ml_result:
        spill_detections = ml_result.get("spill_detections")
        if spill_detections:
            search_region = spill_detections[0].get("polygon_geojson")

    if search_region is None:
        return _error_response(
            422,
            "Unprocessable Entity",
            "No spill region available — run ML inference (and optionally source reconstruction) first",
        )

    # --- Step 4: Determine the historical AIS search window -----------------
    # The window MUST come from the dates the analyst entered for the
    # Sentinel-1 analysis (investigation.start_date / end_date) and the SAR
    # acquisition date of the detection — never from the current wall clock.
    provider_name = (settings.AIS_PROVIDER or "").lower().strip()

    # SAR acquisition / detection instant — the historical moment under
    # investigation (from the Sentinel-1 tile timestamps, falling back to the
    # analyst's entered dates). Used for candidate time-diff scoring.
    detection_time = _acquisition_time(ml_result, investigation)

    # The analyst's entered YYYY-MM-DD range, as a UTC datetime window.
    entered_start, entered_end = _window_from_entered_dates(
        investigation.start_date,
        investigation.end_date,
        padding_days=0,
    )
    if entered_start is None or entered_end is None:
        logger.warning(
            "Attribution: could not parse investigation dates ('%s'..'%s'); "
            "falling back to SAR acquisition date %s ±1 day.",
            investigation.start_date,
            investigation.end_date,
            detection_time.isoformat(),
        )
        entered_start = detection_time - datetime.timedelta(days=1)
        entered_end = detection_time + datetime.timedelta(days=1)

    if provider_name == "gfw":
        # GFW is a historical archive. Search the analyst's entered date range,
        # extended backwards by GFW_LOOKBACK_HOURS so vessels active shortly
        # BEFORE the spill window are captured too (Requirements 2 & 3).
        lookback_hours = max(1, int(settings.GFW_LOOKBACK_HOURS))
        release_window_start = entered_start - datetime.timedelta(hours=lookback_hours)
        release_window_end = entered_end
    else:
        # Other providers: the entered date range, padded by
        # AIS_SEARCH_PADDING_DAYS on each side ("the dates I typed, ±a few days").
        lookback_hours = None
        pad = datetime.timedelta(days=max(0, settings.AIS_SEARCH_PADDING_DAYS))
        release_window_start = entered_start - pad
        release_window_end = entered_end + pad

    # --- Requirement 6 / 7: log the query plan and flag a future window ----
    try:
        region_bbox = shape(search_region).bounds  # (min_lon, min_lat, max_lon, max_lat)
    except Exception:
        region_bbox = None

    now_utc = datetime.datetime.now(timezone.utc)
    diagnostic_warnings: list[str] = []
    if release_window_start > now_utc:
        diagnostic_warnings.append("ais_search_window_in_future")
        logger.error(
            "Attribution: AIS search window %s..%s is entirely in the FUTURE "
            "(now=%s). The investigation's entered dates ('%s'..'%s') look wrong "
            "for a historical search — GFW has no data for this range.",
            release_window_start.isoformat(),
            release_window_end.isoformat(),
            now_utc.isoformat(),
            investigation.start_date,
            investigation.end_date,
        )

    logger.info(
        "Attribution AIS query plan | investigation=%s | provider=%s | "
        "entered_dates=%s..%s | sar_acquisition=%s | search_window=%s..%s | "
        "lookback_hours=%s | region_bbox=%s | buffer_km=%s",
        body.investigation_id,
        provider_name or "(none)",
        investigation.start_date,
        investigation.end_date,
        detection_time.isoformat(),
        release_window_start.isoformat(),
        release_window_end.isoformat(),
        lookback_hours,
        region_bbox,
        settings.AIS_BUFFER_KM,
    )

    # --- Step 5: Query AIS vessels --------------------------------------------
    try:
        ais_result = await ais_service.query_vessels(
            region_geojson=search_region,
            start_time=release_window_start.isoformat(),
            end_time=release_window_end.isoformat(),
            buffer_km=settings.AIS_BUFFER_KM,
        )
    except Exception:
        logger.error(
            "Unexpected error during AIS query:\n%s",
            traceback.format_exc(),
        )
        return _error_response(
            500,
            "Attribution Error",
            "An unexpected error occurred while querying AIS data.",
        )

    logger.info(
        "Attribution AIS result | investigation=%s | provider=%s | "
        "vessels_returned=%d | warnings=%s",
        body.investigation_id,
        provider_name or "(none)",
        len(ais_result.vessels),
        ais_result.warnings,
    )

    # --- Step 6: Filter candidates --------------------------------------------
    try:
        candidates = attribution_service.filter_candidates(
            ais_result=ais_result,
            region_geojson=search_region,
            release_window_start=release_window_start,
            release_window_end=release_window_end,
            buffer_km=settings.AIS_BUFFER_KM,
        )
    except Exception:
        logger.error(
            "Unexpected error during candidate filtering:\n%s",
            traceback.format_exc(),
        )
        return _error_response(
            500,
            "Attribution Error",
            "An unexpected error occurred during candidate filtering.",
        )

    additional_warnings: list[str] = []
    if not candidates:
        # Strict spatial+temporal filter found nothing. Rather than leaving the
        # analyst with zero suspects, fall back to ranking the nearest vessels
        # the AIS feed did return — clearly disclosed via a warning.
        try:
            candidates = attribution_service.filter_candidates_nearest(
                ais_result=ais_result,
                region_geojson=search_region,
                release_window_start=release_window_start,
                release_window_end=release_window_end,
                buffer_km=settings.AIS_BUFFER_KM,
            )
        except Exception:
            logger.error(
                "Unexpected error during relaxed candidate filtering:\n%s",
                traceback.format_exc(),
            )
            candidates = []

        if candidates:
            additional_warnings.append("spatial_filter_relaxed_to_nearest_vessels")
        else:
            additional_warnings.append("no_candidates_found_in_region")

    # --- Step 7: Score and rank candidates ------------------------------------
    try:
        ranked_candidates = attribution_service.score_candidates(
            candidates=candidates,
            source_reconstruction_result=source_recon,
            buffer_km=settings.AIS_BUFFER_KM,
            weights=tuple(attribution_weights),  # type: ignore[arg-type]
            detection_time=detection_time,
            max_distance_km=settings.ATTRIBUTION_MAX_DISTANCE_KM,
            max_time_diff_hours=settings.ATTRIBUTION_MAX_TIME_DIFF_HOURS,
            min_trajectory_overlap=settings.ATTRIBUTION_MIN_TRAJECTORY_OVERLAP,
            min_points_near_source=settings.ATTRIBUTION_MIN_AIS_POINTS_NEAR_SOURCE,
            min_score=settings.ATTRIBUTION_MIN_SCORE,
        )
    except Exception:
        logger.error(
            "Unexpected error during candidate scoring:\n%s",
            traceback.format_exc(),
        )
        return _error_response(
            500,
            "Attribution Error",
            "An unexpected error occurred during candidate scoring.",
        )

    qualified_count = sum(1 for c in ranked_candidates if c.qualified)
    logger.info(
        "Attribution scoring | investigation=%s | scored=%d | qualified=%d "
        "(gates: dist<=%.0fkm time<=%.0fh traj>=%.2f points>=%d score>=%.0f)",
        body.investigation_id,
        len(ranked_candidates),
        qualified_count,
        settings.ATTRIBUTION_MAX_DISTANCE_KM,
        settings.ATTRIBUTION_MAX_TIME_DIFF_HOURS,
        settings.ATTRIBUTION_MIN_TRAJECTORY_OVERLAP,
        settings.ATTRIBUTION_MIN_AIS_POINTS_NEAR_SOURCE,
        settings.ATTRIBUTION_MIN_SCORE,
    )

    # --- Step 8: Build response, persist, and return -------------------------
    try:
        run_at = datetime.datetime.now(timezone.utc)

        gfw_ready = provider_name == "gfw" and bool((settings.GFW_API_TOKEN or "").strip())
        provider_configured = gfw_ready or bool(settings.AIS_API_KEY and settings.AIS_PROVIDER)

        response = AttributionResponse(
            investigation_id=body.investigation_id,
            candidates=ranked_candidates,
            total_candidates=len(ranked_candidates),
            qualified_count=qualified_count,
            ais_provider_configured=provider_configured,
            data_source=provider_name if provider_configured else "none",
            search_region_geojson=search_region,
            release_window_start=release_window_start.isoformat(),
            release_window_end=release_window_end.isoformat(),
            lookback_hours=lookback_hours,
            warnings=ais_result.warnings + additional_warnings + diagnostic_warnings,
            run_at=run_at,
        )

        # Persist result to investigation (Pydantic → plain dict for JSON column)
        investigation.attribution_result = response.model_dump(mode="json")
        await db.commit()

        return {"data": response}

    except Exception:
        logger.error(
            "Unexpected error while building/persisting attribution response:\n%s",
            traceback.format_exc(),
        )
        return _error_response(
            500,
            "Attribution Error",
            "An unexpected error occurred during attribution.",
        )


# ---------------------------------------------------------------------------
# GET /{investigation_id}
# ---------------------------------------------------------------------------


@router.get("/{investigation_id}")
async def get_attribution(
    investigation_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Retrieve a previously stored attribution result for an investigation.

    Returns 404 if the investigation does not exist or if attribution has not
    yet been run.
    """
    investigation = await investigation_service.get_investigation(db, investigation_id)
    if investigation is None:
        return _error_response(404, "Not Found", "Investigation not found.")

    if investigation.attribution_result is None:
        return _error_response(
            404,
            "Not Found",
            "Attribution not yet run for this investigation",
        )

    return {"data": investigation.attribution_result}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _window_from_entered_dates(
    start_date: str | None,
    end_date: str | None,
    padding_days: int,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Build the AIS search window from the investigation's entered dates.

    Returns ``[start_date 00:00 UTC - padding, end_date 23:59:59 UTC + padding]``.
    Both elements are ``None`` when either date string cannot be parsed, so the
    caller can fall back to another strategy.

    Args:
        start_date: ISO date string ``YYYY-MM-DD`` (investigation.start_date).
        end_date: ISO date string ``YYYY-MM-DD`` (investigation.end_date).
        padding_days: Days to extend the window on each side.

    Returns:
        Tuple ``(window_start, window_end)`` as UTC-aware datetimes, or
        ``(None, None)`` on parse failure.
    """
    try:
        pad = datetime.timedelta(days=max(0, padding_days))
        start = datetime.datetime.fromisoformat(str(start_date)).replace(
            tzinfo=timezone.utc
        )
        end = datetime.datetime.fromisoformat(str(end_date)).replace(
            tzinfo=timezone.utc
        )
        # Treat end_date as inclusive — extend to the end of that calendar day.
        end = end + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
        if end < start:
            start, end = end, start
        return start - pad, end + pad
    except (ValueError, AttributeError, TypeError):
        return None, None


def _parse_dt(value: object) -> datetime.datetime | None:
    """Parse an ISO datetime or a bare ``YYYY-MM-DD`` date into a UTC datetime.

    Bare dates are anchored at 12:00:00 UTC (a neutral point in the acquisition
    day). Returns ``None`` when *value* is empty or unparseable.
    """
    if not value:
        return None
    s = str(value).strip()
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        d = datetime.date.fromisoformat(s)
        return datetime.datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None


def _acquisition_time(ml_result: dict | None, investigation) -> datetime.datetime:
    """Return the Sentinel-1 acquisition / detection instant (UTC-aware).

    This is the *historical* moment the analyst is investigating — never the
    time the ML pipeline happened to run. Resolution order:

    1. Latest ``acquisition_date`` among positive (spill) tiles in ``ml_result``.
    2. Latest ``acquisition_date`` among all tiles in ``ml_result``.
    3. The investigation's user-entered ``end_date``.
    4. The investigation's user-entered ``start_date``.
    5. ``ml_result["inference_timestamp"]`` — pipeline run time, last resort.
    6. ``utcnow()`` — fully degraded.
    """
    tiles = (ml_result or {}).get("tile_results") or []
    positive_dates = [
        t.get("acquisition_date")
        for t in tiles
        if isinstance(t, dict) and t.get("is_positive") and t.get("acquisition_date")
    ]
    all_dates = [
        t.get("acquisition_date")
        for t in tiles
        if isinstance(t, dict) and t.get("acquisition_date")
    ]

    ordered_candidates: list[object] = []
    if positive_dates:
        ordered_candidates.append(max(positive_dates))
    if all_dates:
        ordered_candidates.append(max(all_dates))
    ordered_candidates.append(getattr(investigation, "end_date", None))
    ordered_candidates.append(getattr(investigation, "start_date", None))
    ordered_candidates.append((ml_result or {}).get("inference_timestamp"))

    for candidate in ordered_candidates:
        parsed = _parse_dt(candidate)
        if parsed is not None:
            return parsed

    logger.warning(
        "Attribution: no usable SAR acquisition date on the investigation; "
        "falling back to utcnow() (degraded)."
    )
    return datetime.datetime.now(timezone.utc)
