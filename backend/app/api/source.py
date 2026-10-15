"""
FastAPI router for the Source Reconstruction pipeline.

Routes are intentionally thin — all business logic lives in ``drift_service``.
Every successful response is wrapped in ``{"data": ...}`` and every error
is wrapped in ``{"error": ..., "detail": ...}``.

Routes
------
POST /api/source/reconstruct  — trigger kinematic drift backtracking for an investigation
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.source import SourceReconstructionRequest, SourceReconstructionResponse
from app.services import drift_service, investigation_service
from app.services.drift_service import DriftServiceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/source", tags=["source"])


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
# POST /reconstruct
# ---------------------------------------------------------------------------


@router.post("/reconstruct")
async def reconstruct(
    body: SourceReconstructionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Trigger kinematic drift backtracking for an investigation's spill detections.

    Steps
    -----
    1. Look up the investigation; 404 if missing.
    2. Return 409 if reconstruction already exists and ``force_rerun`` is False.
    3. Validate that ``ml_result.spill_detections`` is present and non-empty; 422 otherwise.
    4. Call ``drift_service.compute_source_corridor()``.
    5. On success, persist result to ``investigation.source_reconstruction_result`` and commit.
    6. On ``DriftServiceError``, return the mapped HTTP error without committing.
    7. On any other exception, return 500 without committing.
    """
    # --- Step 1: Validate investigation exists --------------------------------
    investigation = await investigation_service.get_investigation(
        db, body.investigation_id
    )
    if investigation is None:
        return _error_response(404, "Not Found", "Investigation not found.")

    # --- Step 2: Guard against re-runs without force_rerun --------------------
    if investigation.source_reconstruction_result and not body.force_rerun:
        return _error_response(
            409,
            "Conflict",
            "Source reconstruction already completed. Pass force_rerun=true to re-run.",
        )

    # --- Step 3: Validate ML result and spill detections ----------------------
    ml_result = investigation.ml_result
    if not ml_result or not ml_result.get("spill_detections"):
        return _error_response(
            422,
            "Unprocessable Entity",
            "No spill detections found in investigation — run ML inference first",
        )

    spill_detections: list[dict] = ml_result["spill_detections"]

    # Derive acquisition timestamp from the stored inference_timestamp.
    # Fall back to utcnow() only when the field is absent (degraded mode).
    raw_ts = ml_result.get("inference_timestamp")
    if raw_ts:
        try:
            acquisition_timestamp = datetime.datetime.fromisoformat(
                raw_ts.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            logger.warning(
                "Could not parse inference_timestamp '%s'; falling back to utcnow.", raw_ts
            )
            acquisition_timestamp = datetime.datetime.now(datetime.timezone.utc)
    else:
        acquisition_timestamp = datetime.datetime.now(datetime.timezone.utc)

    # --- Step 4–7: Run reconstruction, persist, and return -------------------
    try:
        output = await drift_service.compute_source_corridor(
            spill_detections=spill_detections,
            acquisition_timestamp=acquisition_timestamp,
            drift_speed_m_s=body.drift_speed_m_s,
            lookback_hours=body.lookback_hours,
        )

        # Persist result (dataclass → plain dict for JSON column storage)
        run_at = datetime.datetime.now(datetime.timezone.utc)
        result_dict = dataclasses.asdict(output)
        # datetime fields are serialised as ISO strings for JSON storage
        result_dict["release_window_start"] = output.release_window_start.isoformat()
        result_dict["release_window_end"] = output.release_window_end.isoformat()
        result_dict["investigation_id"] = body.investigation_id
        result_dict["run_at"] = run_at.isoformat()

        investigation.source_reconstruction_result = result_dict
        await db.commit()

        return {
            "data": SourceReconstructionResponse(
                investigation_id=body.investigation_id,
                source_corridor_geojson=output.source_corridor_geojson,
                release_window_start=output.release_window_start.isoformat(),
                release_window_end=output.release_window_end.isoformat(),
                source_confidence=output.source_confidence,
                reconstruction_method=output.reconstruction_method,
                drift_distance_km=output.drift_distance_km,
                warnings=output.warnings,
                run_at=run_at,
            )
        }

    except DriftServiceError as exc:
        logger.error("DriftServiceError during reconstruction: %s", exc.message)
        return _error_response(
            exc.status_code,
            "Source Reconstruction Error",
            exc.message,
        )

    except Exception:
        logger.error(
            "Unexpected error during source reconstruction:\n%s",
            traceback.format_exc(),
        )
        return _error_response(
            500,
            "Source Reconstruction Error",
            "An unexpected error occurred during source reconstruction.",
        )
