"""
ORICEAN FastAPI application entry point.

Responsibilities
----------------
- FastAPI lifespan: startup initialization and shutdown cleanup
- Router registration: investigations and ML inference
- Middleware: CORS
- Global exception handler: catches unhandled exceptions, logs full
  traceback, and returns a consistent HTTP 500 response envelope
- Health endpoint: reflects actual runtime state (model + DB)
"""

from __future__ import annotations

import logging
import pathlib
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import alerts, investigations, ml, officer, public, source, attribution
from app.core.config import settings
from app.db.database import engine, Base
from app.models import investigation as _investigation_model  # noqa: F401 — registers with Base
from app.models import alert as _alert_model  # noqa: F401 — registers Alert + NotificationSettings with Base
from app.models import officer_case as _officer_case_model  # noqa: F401 — registers OfficerCase with Base
from ml.inference.predictor import OilSpillPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

# Hugging Face repo the SAR oil-slick weights are pulled from when MODEL_PATH
# is missing on disk (the 94 MB model.pth is git-ignored, so a fresh clone /
# pull never carries it — see backend/.gitignore).
_HF_MODEL_REPO_ID = "MeghanaK25/sar-oil-slick-detection"
_HF_MODEL_FILENAME = "model.pth"


def _ensure_model_weights() -> None:
    """Download the model weights from Hugging Face when they are not on disk.

    No-op when ``settings.MODEL_PATH`` already exists. On any failure this logs
    a warning and returns — the lifespan then continues into degraded mode
    exactly as before (health reports ``model_loaded=False``; inference 503s).
    """
    if not settings.MODEL_PATH:
        return

    target = pathlib.Path(settings.MODEL_PATH)
    if not target.is_absolute():
        # MODEL_PATH is repo-relative to the backend/ working directory.
        target = pathlib.Path.cwd() / target

    if target.is_file() and target.stat().st_size > 0:
        return

    logger.info(
        "Startup: model weights missing at '%s' — downloading '%s' from "
        "Hugging Face (%s)…",
        target,
        _HF_MODEL_FILENAME,
        _HF_MODEL_REPO_ID,
    )
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        downloaded = hf_hub_download(
            repo_id=_HF_MODEL_REPO_ID,
            filename=_HF_MODEL_FILENAME,
            token=settings.HF_TOKEN or None,
            local_dir=str(target.parent),
        )
        logger.info("Startup: model weights downloaded to '%s'.", downloaded)
    except Exception as exc:
        logger.warning(
            "Startup: could not download model weights (%s). Inference will "
            "run in degraded mode until '%s' is present.",
            exc,
            target,
        )

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup sequence
    ----------------
    1. Store a reference to the shared DB engine on ``app.state`` and probe
       connectivity with a ``SELECT 1``.
    2. Authenticate with Google Earth Engine (skipped when GEE_MOCK_MODE=True
       or when no credentials are configured).
    3. Load the OilSpillPredictor from the configured model path.

    Shutdown
    --------
    - Dispose the async DB engine connection pool.
    """

    # ------------------------------------------------------------------
    # 1. Database
    # ------------------------------------------------------------------
    logger.info("Startup: initializing database connection pool…")
    app.state.db_engine = engine
    app.state.db_connected = False

    try:
        # Create tables if they don't exist yet (idempotent)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Startup: database tables verified/created.")

        async with AsyncSession(engine) as session:
            await session.execute(text("SELECT 1"))
        app.state.db_connected = True
        logger.info("Startup: database connectivity confirmed.")
    except Exception as exc:  # pragma: no cover
        logger.error("Startup: database connectivity check failed: %s", exc)
        # App continues — health endpoint will reflect db_connected=False

    # ------------------------------------------------------------------
    # 2. Google Earth Engine authentication
    # ------------------------------------------------------------------
    logger.info("Startup: authenticating Google Earth Engine…")

    if settings.GEE_MOCK_MODE:
        logger.info(
            "Startup: GEE_MOCK_MODE=True — skipping real GEE authentication."
        )
    else:
        credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS.strip()
        project_id = settings.GEE_PROJECT_ID.strip() or None

        if not credentials_path:
            logger.info(
                "Startup: GOOGLE_APPLICATION_CREDENTIALS is empty — "
                "skipping GEE authentication (running without live GEE)."
            )
        else:
            try:
                import ee  # noqa: PLC0415
                import google.oauth2.service_account as sa  # type: ignore

                gee_credentials = sa.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/earthengine"],
                )
                ee.Initialize(credentials=gee_credentials, project=project_id)
                logger.info(
                    "Startup: GEE authenticated with service-account credentials."
                )
            except Exception as exc:
                logger.error(
                    "Startup: GEE authentication failed: %s. "
                    "Satellite imagery endpoints will be unavailable.",
                    exc,
                )
                raise SystemExit(1) from exc

    # ------------------------------------------------------------------
    # 3. ML model
    # ------------------------------------------------------------------
    logger.info("Startup: loading OilSpillPredictor (model_path=%s)…", settings.MODEL_PATH)

    _ensure_model_weights()

    predictor = OilSpillPredictor(
        model_path=settings.MODEL_PATH,
        device=settings.INFERENCE_DEVICE,
        config_path=settings.MODEL_CONFIG_PATH or None,
    )

    try:
        predictor.load()
        logger.info(
            "Startup: ML model loaded successfully (version=%s, device=%s).",
            predictor.model_version,
            settings.INFERENCE_DEVICE,
        )
    except FileNotFoundError as exc:
        logger.error(
            "Startup: ML model file not found at '%s'. "
            "Inference endpoints will return 503 until the model is present. "
            "Error: %s",
            settings.MODEL_PATH,
            exc,
        )
        # Do NOT exit — app runs in degraded mode; health will show model_loaded=False
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Startup: unexpected error while loading ML model: %s", exc
        )
        # Same degraded-mode approach

    app.state.predictor = predictor

    # ------------------------------------------------------------------
    # Yield control to FastAPI — application is now running
    # ------------------------------------------------------------------
    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Shutdown: disposing database engine…")
    await app.state.db_engine.dispose()
    logger.info("Shutdown: complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ORICEAN API",
    description="AI-powered oil spill intelligence platform",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled exceptions in any route.

    Logs the full traceback at ERROR level and returns a generic HTTP 500
    response that does not expose internal implementation details.
    """
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

# investigations.router has no prefix itself; mount it at /investigations
app.include_router(investigations.router, prefix="/investigations")

# ml.router already declares prefix="/api/ml" — include without extra prefix
app.include_router(ml.router)

# source.router declares prefix="/api/source" — include without extra prefix
app.include_router(source.router)

# attribution.router declares prefix="/api/attribution" — include without extra prefix
app.include_router(attribution.router)

# alerts.router declares prefix="/api/alerts" — officer SMS alerting
app.include_router(alerts.router)

# public.router declares prefix="/api/public" — sanitised public incident feed
app.include_router(public.router)

# officer.router declares prefix="/api/officer" — officer case workflow
app.include_router(officer.router)

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    """Simple liveness probe — confirms the API process is reachable."""
    return {"message": "ORICEAN Backend Running"}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(request: Request) -> dict:
    """
    Return the runtime health of the application.

    Response shape
    --------------
    ``{"status": "healthy", "model_loaded": <bool>, "db_connected": <bool>}``

    ``model_loaded`` is read from ``app.state.predictor.is_loaded``.
    ``db_connected`` is read from ``app.state.db_connected``.
    """
    predictor: OilSpillPredictor = request.app.state.predictor
    db_connected: bool = getattr(request.app.state, "db_connected", False)

    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded,
        "db_connected": db_connected,
    }
