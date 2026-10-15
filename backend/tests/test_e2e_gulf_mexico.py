"""
Final end-to-end checkpoint test — Gulf of Mexico AOI (mock mode).

Verifies:
- POST /investigations  →  creates a new investigation (201)
- POST /api/ml/run-inference  →  returns InferenceResult JSON with
  mock_mode=true and at least one SpillDetection
- GET /api/ml/health  →  status is "mock_mode" or "ready"

All tests run with ML_MOCK_MODE=true and GEE_MOCK_MODE=true.
No real model weights or GEE credentials are required.
"""

import os
import random

# Set mock env vars BEFORE any app imports
os.environ["ML_MOCK_MODE"] = "true"
os.environ["GEE_MOCK_MODE"] = "true"
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
os.environ.setdefault(
    "MODEL_PATH",
    "ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
)
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")
os.environ.setdefault("INFERENCE_DEVICE", "cpu")

import asyncio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base, get_db
from app.main import app
from app.services.ml_service import get_predictor
from ml.inference.predictor import OilSpillPredictor

# ---------------------------------------------------------------------------
# Gulf of Mexico AOI (task requirement)
# ---------------------------------------------------------------------------

GULF_AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [-90.5, 28.5],
            [-89.5, 28.5],
            [-89.5, 29.5],
            [-90.5, 29.5],
            [-90.5, 28.5],
        ]
    ],
}

# ---------------------------------------------------------------------------
# App fixture — in-memory SQLite + unloaded predictor (mock mode only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """TestClient with in-memory SQLite DB and mock predictor."""
    predictor = OilSpillPredictor(
        model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
        device="cpu",
    )
    # Do NOT call predictor.load() — mock mode does not need weights.
    app.dependency_overrides[get_predictor] = lambda: predictor

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async def _setup():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    test_session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )

    async def override_get_db():
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()

    async def _teardown():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await test_engine.dispose()

    asyncio.run(_teardown())


# ---------------------------------------------------------------------------
# Test 1 — GET /api/ml/health
# ---------------------------------------------------------------------------


def test_health_returns_mock_mode(client):
    """
    GET /api/ml/health must return status 'mock_mode' when ML_MOCK_MODE=true.
    """
    resp = client.get("/api/ml/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "data" in body, f"Response missing 'data' key: {body}"
    status = body["data"]["status"]
    assert status in (
        "mock_mode",
        "ready",
    ), f"Expected 'mock_mode' or 'ready', got: {status!r}"
    # In ML_MOCK_MODE we specifically expect "mock_mode"
    assert status == "mock_mode", (
        f"ML_MOCK_MODE=true should produce status='mock_mode', got {status!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — POST /api/ml/run-inference with Gulf of Mexico AOI
# ---------------------------------------------------------------------------


def test_run_inference_gulf_mexico(client):
    """
    Full pipeline test with Gulf of Mexico AOI.

    1. Create an investigation.
    2. Run inference with Gulf AOI.
    3. Assert InferenceResult contains mock_mode=true and >= 1 SpillDetection.
    """
    # Step 1: Create investigation
    random.seed(42)  # deterministic mock confidences
    create_resp = client.post(
        "/investigations/",
        json={
            "name": "Gulf of Mexico Checkpoint",
            "aoi_geojson": GULF_AOI,
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
        },
    )
    assert create_resp.status_code == 201, (
        f"Investigation creation failed ({create_resp.status_code}): {create_resp.text}"
    )
    inv_id = create_resp.json()["data"]["id"]
    assert inv_id, "Investigation ID must not be empty"

    # Step 2: Run ML inference
    infer_resp = client.post(
        "/api/ml/run-inference",
        json={
            "investigation_id": inv_id,
            "aoi_geojson": GULF_AOI,
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            # Low threshold to guarantee >= 1 positive from 3 random mock tiles
            "confidence_threshold": 0.1,
        },
    )
    assert infer_resp.status_code == 200, (
        f"run-inference failed ({infer_resp.status_code}): {infer_resp.text}"
    )

    # Step 3: Validate InferenceResult shape
    body = infer_resp.json()
    assert "data" in body, f"Response missing 'data' key: {body}"
    result = body["data"]

    # mock_mode flag
    assert result.get("mock_mode") is True, (
        f"Expected mock_mode=true, got: {result.get('mock_mode')}"
    )

    # tile count — sliding window on 1 mock 512×512 raster produces many crops,
    # confirming the fix (old code always returned exactly 3 tiles).
    assert result["total_tiles_processed"] > 4, (
        f"Expected > 4 tiles (sliding window crops from 512×512 raster), got {result['total_tiles_processed']}"
    )

    # at least one SpillDetection
    assert result["positive_tile_count"] >= 1, (
        f"Expected >= 1 positive tile, got {result['positive_tile_count']}"
    )
    detections = result["spill_detections"]
    assert len(detections) >= 1, (
        f"Expected >= 1 SpillDetection, got {len(detections)}"
    )

    # SpillDetection structure
    det = detections[0]
    assert det["detection_id"], "detection_id must not be empty"
    assert det["polygon_geojson"], "polygon_geojson must not be empty"
    assert "type" in det["polygon_geojson"], "polygon_geojson must have a 'type' key"
    assert det["tile_count"] >= 1
    assert 0.0 <= det["mean_confidence"] <= 1.0
    assert 0.0 <= det["max_confidence"] <= 1.0

    # investigation_id preserved
    assert result["investigation_id"] == inv_id

    # All tile bounds must be subsets of the Gulf AOI bounding box (-90.5, 28.5, -89.5, 29.5)
    gulf_min_lon, gulf_min_lat, gulf_max_lon, gulf_max_lat = -90.5, 28.5, -89.5, 29.5
    tile_results = result["tile_results"]
    for i, tile in enumerate(tile_results):
        bounds = tile["bounds"]
        assert bounds[0] >= gulf_min_lon, (
            f"Tile {i} min_lon {bounds[0]} < AOI min_lon {gulf_min_lon}"
        )
        assert bounds[1] >= gulf_min_lat, (
            f"Tile {i} min_lat {bounds[1]} < AOI min_lat {gulf_min_lat}"
        )
        assert bounds[2] <= gulf_max_lon, (
            f"Tile {i} max_lon {bounds[2]} > AOI max_lon {gulf_max_lon}"
        )
        assert bounds[3] <= gulf_max_lat, (
            f"Tile {i} max_lat {bounds[3]} > AOI max_lat {gulf_max_lat}"
        )

    # At least two tiles must have different bounds (per-crop bounds are distinct)
    assert len(tile_results) >= 2, (
        "Expected at least 2 tile results to verify distinct bounds"
    )
    all_bounds = [tuple(t["bounds"]) for t in tile_results]
    unique_bounds = set(all_bounds)
    assert len(unique_bounds) > 1, (
        f"Expected distinct per-crop bounds, but all {len(all_bounds)} tiles share the same bounds: {all_bounds[0]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — investigation transitions to 'complete' after inference
# ---------------------------------------------------------------------------


def test_investigation_status_complete_after_inference(client):
    """After a successful run-inference call the investigation status must be 'complete'."""
    random.seed(42)
    create_resp = client.post(
        "/investigations/",
        json={
            "name": "Gulf Status Check",
            "aoi_geojson": GULF_AOI,
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
        },
    )
    assert create_resp.status_code == 201
    inv_id = create_resp.json()["data"]["id"]

    client.post(
        "/api/ml/run-inference",
        json={
            "investigation_id": inv_id,
            "aoi_geojson": GULF_AOI,
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "confidence_threshold": 0.1,
        },
    )

    get_resp = client.get(f"/investigations/{inv_id}")
    assert get_resp.status_code == 200
    inv_data = get_resp.json()["data"]
    assert inv_data["status"] == "complete", (
        f"Expected status='complete', got {inv_data['status']!r}"
    )
    assert inv_data["ml_result"] is not None, "ml_result should be persisted after inference"
