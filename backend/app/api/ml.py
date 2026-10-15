"""
FastAPI router for the ML inference pipeline.

Routes are intentionally thin — all business logic lives in ``ml_service``.
Every successful response is wrapped in ``{"data": ...}`` and every error
is wrapped in ``{"error": ..., "detail": ...}``.

Routes
------
POST /run-inference  — trigger ML inference for an investigation
GET  /health         — ML model readiness check
GET  /model-info     — static metadata about the loaded model architecture
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.models.investigation import InvestigationStatus
from app.schemas.ml import (
    MLHealthResponse,
    MLInferenceRequest,
    ModelInfoResponse,
)
from app.services import investigation_service
from app.services import ml_service
from app.services.ml_service import MLServiceError, get_predictor
from ml.inference.predictor import OilSpillPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata file — loaded once at module import
# ---------------------------------------------------------------------------

_METADATA_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "ml"
    / "models"
    / "model_metadata.json"
)

try:
    with _METADATA_PATH.open() as _f:
        _MODEL_METADATA: dict = json.load(_f)
except FileNotFoundError:
    logger.warning("model_metadata.json not found at %s", _METADATA_PATH)
    _MODEL_METADATA = {}

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/ml", tags=["ml"])


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
# POST /run-inference
# ---------------------------------------------------------------------------


@router.post("/run-inference")
async def run_inference(
    request: MLInferenceRequest,
    db: AsyncSession = Depends(get_db),
    predictor: OilSpillPredictor = Depends(get_predictor),
) -> dict:
    """
    Trigger the full SAR → ML inference pipeline for an investigation.

    Steps
    -----
    1. Look up the investigation; 404 if missing.
    2. Guard against re-running on an already-processing or completed record.
    3. Set status to ``running_ml``.
    4. Run ``ml_service.run_ml_inference``.
    5. On success, persist the result and set status to ``complete``.
    6. On ``MLServiceError``, set status to ``failed`` and surface the error.
    7. On any other exception, set status to ``failed`` and return 500.
    """
    # --- Step 1: Validate investigation exists ---
    investigation = await investigation_service.get_investigation(
        db, request.investigation_id
    )
    if investigation is None:
        return _error_response(404, "Not Found", "Investigation not found.")

    # --- Step 2: Guard against re-runs ---
    if investigation.status in (
        InvestigationStatus.running_ml,
        InvestigationStatus.complete,
    ):
        return _error_response(
            409,
            "Conflict",
            "Investigation is already processing or completed.",
        )

    # --- Step 3: Mark as running ---
    await investigation_service.set_status(
        db, request.investigation_id, InvestigationStatus.running_ml
    )

    try:
        # --- Step 4: Run inference ---
        result = await ml_service.run_ml_inference(request, predictor)

        # --- Step 5: Persist result and mark complete ---
        await investigation_service.set_status(
            db, request.investigation_id, InvestigationStatus.complete
        )

        # Write ml_result and ml_run_at directly on the ORM object
        investigation = await investigation_service.get_investigation(
            db, request.investigation_id
        )
        investigation.ml_result = result.model_dump(mode="json")
        investigation.ml_run_at = datetime.datetime.utcnow()
        await db.commit()

        # --- Step 5b: Officer alert on a significant detection (best-effort) ---
        # Never let alerting failure affect the inference response.
        try:
            from app.services import alert_service

            await alert_service.maybe_create_and_send_alert(
                db, investigation, investigation.ml_result
            )
        except Exception:
            logger.warning(
                "Detection alert dispatch failed (ignored):\n%s", traceback.format_exc()
            )

        return {"data": result}

    except MLServiceError as exc:
        # --- Step 6: Known ML service error ---
        logger.error("MLServiceError during inference: %s", exc.message)
        await investigation_service.set_status(
            db,
            request.investigation_id,
            InvestigationStatus.failed,
            error_message=str(exc),
        )
        return _error_response(exc.status_code, "ML Service Error", exc.message)

    except Exception:
        # --- Step 7: Unexpected error ---
        logger.error(
            "Unexpected error during ML inference:\n%s", traceback.format_exc()
        )
        await investigation_service.set_status(
            db,
            request.investigation_id,
            InvestigationStatus.failed,
            error_message="Unexpected internal error during ML inference.",
        )
        return _error_response(
            500,
            "Internal Server Error",
            "An unexpected error occurred during ML inference.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(
    predictor: OilSpillPredictor = Depends(get_predictor),
) -> dict:
    """
    Return the current readiness status of the ML model.

    Returns
    -------
    dict
        ``{"data": MLHealthResponse}``
    """
    try:
        if settings.ML_MOCK_MODE:
            status = "mock_mode"
            model_loaded = True
        elif predictor.is_loaded:
            status = "ready"
            model_loaded = True
        else:
            status = "not_loaded"
            model_loaded = False

        model_version = predictor.model_version if predictor.is_loaded else None

        return {
            "data": MLHealthResponse(
                status=status,
                model_loaded=model_loaded,
                model_version=model_version,
                device=settings.INFERENCE_DEVICE,
            )
        }
    except Exception:
        logger.error("Unexpected error in /health:\n%s", traceback.format_exc())
        return _error_response(
            500, "Internal Server Error", "Failed to retrieve ML health status."
        )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@router.get("/model-info")
async def model_info(
    predictor: OilSpillPredictor = Depends(get_predictor),
) -> dict:
    """
    Return static metadata about the loaded model architecture.

    Reads ``model_metadata.json`` (loaded once at module import) and
    returns it as a ``ModelInfoResponse``.

    Returns
    -------
    dict
        ``{"data": ModelInfoResponse}``
    """
    try:
        # Determine which variant is active from the model version string
        model_version: str | None = (
            predictor.model_version if predictor.is_loaded else None
        )
        if model_version and "student" in model_version.lower():
            model_variant = "student"
        else:
            model_variant = "primary"

        # The JSON stores [H, W]; the schema expects [C, H, W]
        raw_input_size: list[int] = _MODEL_METADATA.get("input_size", [224, 224])
        input_size_chw = [3, raw_input_size[0], raw_input_size[1]]

        return {
            "data": ModelInfoResponse(
                architecture=_MODEL_METADATA.get("architecture", "unknown"),
                num_classes=_MODEL_METADATA.get("num_classes", 2),
                class_names=_MODEL_METADATA.get("class_names", []),
                input_size=input_size_chw,
                normalization=_MODEL_METADATA.get("normalization", {}),
                model_variant=model_variant,
            )
        }
    except Exception:
        logger.error("Unexpected error in /model-info:\n%s", traceback.format_exc())
        return _error_response(
            500, "Internal Server Error", "Failed to retrieve model info."
        )
