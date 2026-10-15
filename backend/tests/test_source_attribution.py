"""
Unit tests for the Source Reconstruction and Vessel Attribution pipelines.

Covers:
- POST /api/source/reconstruct  (Tasks 12.1 tests 1, 2)
- drift_service.compute_source_corridor  (Tasks 12.1 tests 3, 4)
- ais_service.query_vessels  (Task 12.2 test 5)
- attribution_service.filter_candidates  (Task 12.2 test 6)
- POST /api/attribution/run  (Task 12.2 test 7)
- config._parse_attribution_weights  (Task 12.2 test 8)

Infrastructure mirrors test_investigation_service.py:
- Stub env-vars set before any app imports
- pytest-asyncio for async tests
- In-memory SQLite (aiosqlite) for route tests
- httpx.AsyncClient with ASGITransport for HTTP route tests
"""

import os
import uuid

# ---------------------------------------------------------------------------
# Stub env-vars BEFORE any app module is imported
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
os.environ.setdefault("MODEL_PATH", "/dev/null")
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")

# ---------------------------------------------------------------------------
# Standard library + third-party imports
# ---------------------------------------------------------------------------
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# App imports (safe after env stubs)
# ---------------------------------------------------------------------------
from app.core.config import _parse_attribution_weights
from app.db.database import Base, get_db
from app.main import app
from app.models.investigation import Investigation, InvestigationStatus
from app.schemas.attribution import AISQueryResult
from app.services import ais_service, attribution_service, drift_service

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

_SAMPLE_AOI = {
    "type": "Polygon",
    "coordinates": [[[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]],
}

_SAMPLE_POLYGON_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]],
}

_VALID_SPILL_DETECTIONS = [
    {
        "polygon_geojson": _SAMPLE_POLYGON_GEOJSON,
        "mean_confidence": 0.85,
        "area_km2": 10.0,
    }
]

_VALID_ML_RESULT = {
    "spill_detections": _VALID_SPILL_DETECTIONS,
    "inference_timestamp": "2024-01-15T12:00:00Z",
}

_ACQUISITION_TIMESTAMP = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Async DB fixture (in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    """Yield a fresh AsyncSession backed by an in-memory SQLite database."""
    engine = create_async_engine(TEST_DB_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ---------------------------------------------------------------------------
# HTTP client fixture (overrides the DB dependency with the test DB)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(db: AsyncSession):
    """
    Yield an httpx.AsyncClient wired to the FastAPI app.

    The ``get_db`` dependency is overridden so all route handlers use the
    same in-memory SQLite session as the test.
    """

    async def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper: insert a minimal Investigation row directly via ORM
# ---------------------------------------------------------------------------


async def _create_investigation(
    db: AsyncSession,
    *,
    ml_result=None,
    source_reconstruction_result=None,
    attribution_result=None,
) -> Investigation:
    """Insert a minimal Investigation into the test DB and return it."""
    inv = Investigation(
        id=str(uuid.uuid4()),
        name="Test Investigation",
        aoi_geojson=_SAMPLE_AOI,
        start_date="2024-01-01",
        end_date="2024-01-31",
        status=InvestigationStatus.pending,
        ml_result=ml_result,
        source_reconstruction_result=source_reconstruction_result,
        attribution_result=attribution_result,
    )
    db.add(inv)
    await db.commit()
    await db.refresh(inv)
    return inv


# ===========================================================================
# Task 12.1 — Tests 1 & 2: Source reconstruction route validation
# ===========================================================================


@pytest.mark.asyncio
async def test_reconstruct_returns_422_when_no_ml_result(client, db):
    """POST /api/source/reconstruct with ml_result=None must return HTTP 422
    and the error detail must mention 'run ML inference first'."""
    inv = await _create_investigation(db, ml_result=None)

    response = await client.post(
        "/api/source/reconstruct",
        json={"investigation_id": inv.id},
    )

    assert response.status_code == 422
    body = response.json()
    assert "run ML inference first" in body.get("detail", "")


@pytest.mark.asyncio
async def test_reconstruct_returns_422_when_empty_detections(client, db):
    """POST /api/source/reconstruct with spill_detections=[] must return HTTP 422."""
    inv = await _create_investigation(
        db,
        ml_result={"spill_detections": [], "inference_timestamp": "2024-01-15T12:00:00Z"},
    )

    response = await client.post(
        "/api/source/reconstruct",
        json={"investigation_id": inv.id},
    )

    assert response.status_code == 422


# ===========================================================================
# Task 12.1 — Tests 3 & 4: drift_service service-level tests
# ===========================================================================


@pytest.mark.asyncio
async def test_reconstruction_method_field_value():
    """compute_source_corridor() must set reconstruction_method to 'kinematic_drift_mvp'."""
    output = await drift_service.compute_source_corridor(
        spill_detections=_VALID_SPILL_DETECTIONS,
        acquisition_timestamp=_ACQUISITION_TIMESTAMP,
    )
    assert output.reconstruction_method == "kinematic_drift_mvp"


@pytest.mark.asyncio
async def test_no_ocean_current_warning_present():
    """compute_source_corridor() must include 'no_ocean_current_data_used' in warnings."""
    output = await drift_service.compute_source_corridor(
        spill_detections=_VALID_SPILL_DETECTIONS,
        acquisition_timestamp=_ACQUISITION_TIMESTAMP,
    )
    assert "no_ocean_current_data_used" in output.warnings


# ===========================================================================
# Task 12.2 — Test 5: ais_service with no provider configured
# ===========================================================================


@pytest.mark.asyncio
async def test_ais_service_returns_empty_when_no_provider(monkeypatch):
    """query_vessels() must return vessels=[] and warnings=['no_ais_provider_configured']
    when AIS_API_KEY and AIS_PROVIDER are both empty, without making any outbound request."""
    monkeypatch.setattr(ais_service.settings, "AIS_API_KEY", "")
    monkeypatch.setattr(ais_service.settings, "AIS_PROVIDER", "")

    result = await ais_service.query_vessels(
        region_geojson=_SAMPLE_POLYGON_GEOJSON,
        start_time="2024-01-13T12:00:00Z",
        end_time="2024-01-15T12:00:00Z",
        buffer_km=20.0,
    )

    assert result.vessels == []
    assert result.warnings == ["no_ais_provider_configured"]


# ===========================================================================
# Task 12.2 — Test 6: filter_candidates with empty AIS result
# ===========================================================================


def test_filter_candidates_empty_input_returns_empty_list():
    """filter_candidates() with an empty AISQueryResult must return [] without raising."""
    rw_start = datetime(2024, 1, 13, 12, 0, 0, tzinfo=timezone.utc)
    rw_end = datetime(2024, 1, 14, 12, 0, 0, tzinfo=timezone.utc)

    result = attribution_service.filter_candidates(
        ais_result=AISQueryResult(vessels=[], warnings=[]),
        region_geojson=_SAMPLE_POLYGON_GEOJSON,
        release_window_start=rw_start,
        release_window_end=rw_end,
        buffer_km=20.0,
    )

    assert result == []


# ===========================================================================
# Task 12.2 — Test 7: attribution route when AIS is unconfigured
# ===========================================================================


@pytest.mark.asyncio
async def test_attribution_returns_no_provider_false_when_unconfigured(
    client, db, monkeypatch
):
    """POST /api/attribution/run must return ais_provider_configured=false and
    candidates=[] when AIS credentials are not set."""
    # Patch settings on both the ais_service module and the attribution router
    monkeypatch.setattr(ais_service.settings, "AIS_API_KEY", "")
    monkeypatch.setattr(ais_service.settings, "AIS_PROVIDER", "")

    # Also patch the attribution router's copy of settings
    import app.api.attribution as attribution_router_module
    monkeypatch.setattr(attribution_router_module.settings, "AIS_API_KEY", "")
    monkeypatch.setattr(attribution_router_module.settings, "AIS_PROVIDER", "")

    inv = await _create_investigation(db, ml_result=_VALID_ML_RESULT)

    response = await client.post(
        "/api/attribution/run",
        json={"investigation_id": inv.id},
    )

    assert response.status_code == 200
    body = response.json()
    data = body.get("data", {})
    assert data.get("ais_provider_configured") is False
    assert data.get("candidates") == []


# ===========================================================================
# Task 12.2 — Test 8: _parse_attribution_weights reverts to default on invalid input
# ===========================================================================


def test_attribution_weights_loader_invalid_reverts_to_default():
    """_parse_attribution_weights with only 2 values must return the default [0.35, 0.30, 0.25, 0.10]."""
    result = _parse_attribution_weights("0.5,0.5")
    assert result == [0.35, 0.30, 0.25, 0.10]


# ===========================================================================
# GFW date-window regression: the AIS search MUST use the investigation's
# previously entered historical dates, never the current wall clock.
# ===========================================================================


@pytest.mark.asyncio
async def test_gfw_attribution_window_uses_entered_investigation_dates(
    client, db, monkeypatch
):
    """POST /api/attribution/run with AIS_PROVIDER=gfw must query AIS for the
    investigation's entered 2024 date range (extended backwards by the GFW
    lookback), NOT a window anchored to today's date."""
    import app.api.attribution as attribution_router_module

    monkeypatch.setattr(attribution_router_module.settings, "AIS_PROVIDER", "gfw")
    monkeypatch.setattr(attribution_router_module.settings, "GFW_API_TOKEN", "test-token")
    monkeypatch.setattr(attribution_router_module.settings, "GFW_LOOKBACK_HOURS", 48)

    captured: dict[str, str] = {}

    async def _capture_query_vessels(*, region_geojson, start_time, end_time, buffer_km):
        captured["start_time"] = start_time
        captured["end_time"] = end_time
        return AISQueryResult(vessels=[], warnings=["gfw_historical_ais"])

    monkeypatch.setattr(
        attribution_router_module.ais_service, "query_vessels", _capture_query_vessels
    )

    ml_result = {
        "spill_detections": _VALID_SPILL_DETECTIONS,
        # inference_timestamp is "now" in production — it must be ignored here.
        "inference_timestamp": "2026-09-08T10:00:00Z",
        "tile_results": [
            {"is_positive": True, "acquisition_date": "2024-01-14"},
            {"is_positive": False, "acquisition_date": "2024-01-10"},
        ],
    }
    inv = await _create_investigation(db, ml_result=ml_result)  # 2024-01-01 .. 2024-01-31

    response = await client.post(
        "/api/attribution/run", json={"investigation_id": inv.id}
    )
    assert response.status_code == 200

    start = datetime.fromisoformat(captured["start_time"])
    end = datetime.fromisoformat(captured["end_time"])

    # Historical entered range — anchored to the 2024 dates, NOT today's date.
    # The 48 h GFW lookback pushes the start just before 2024-01-01.
    assert datetime(2023, 12, 29, tzinfo=timezone.utc) <= start < datetime(2024, 1, 1, tzinfo=timezone.utc)
    # End stays at the entered end_date (inclusive to end-of-day).
    assert datetime(2024, 1, 31, tzinfo=timezone.utc) <= end < datetime(2024, 2, 2, tzinfo=timezone.utc)

    data = response.json()["data"]
    assert data["release_window_start"].startswith("2023-12-30")
    assert data["release_window_end"].startswith("2024-01-31")
    assert "ais_search_window_in_future" not in data["warnings"]
