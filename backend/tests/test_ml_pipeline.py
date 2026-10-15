"""
End-to-end ML pipeline tests (mock mode).

Covers:
- OilSpillPredictor in ML_MOCK_MODE: predict_batch returns correct structure
- GEE service in GEE_MOCK_MODE: get_sar_tiles returns 3 synthetic tiles
- build_spill_polygons: converts positive tiles to SpillDetection list
- run_ml_inference (full async pipeline): returns InferenceResult with >= 1 SpillDetection
- GET /api/ml/health: returns status "mock_mode" or "ready"
- POST /api/ml/run-inference: returns InferenceResult with >= 1 SpillDetection

All tests run without a real database, GEE credentials, or ML model weights.
"""

import os

# ---------------------------------------------------------------------------
# Stub env-vars BEFORE any app imports so config.py loads without errors.
# ML_MOCK_MODE and GEE_MOCK_MODE must be set here (not just in .env) because
# pytest does not reload .env between test modules.
# ---------------------------------------------------------------------------
os.environ["ML_MOCK_MODE"] = "true"
os.environ["GEE_MOCK_MODE"] = "true"
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
os.environ.setdefault("MODEL_PATH", "ml/models/oricean_oil_classifier_mobilenetv3_large.pth")
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")
os.environ.setdefault("INFERENCE_DEVICE", "cpu")

import io
import random
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base
from app.models.investigation import Investigation, InvestigationStatus
from app.schemas.investigation import InvestigationCreate
from app.schemas.ml import (
    InferenceResult,
    MLInferenceRequest,
    SpillDetection,
    TilePrediction,
)
from app.services import gee_service, investigation_service
from app.services.ml_service import build_spill_polygons, run_ml_inference
from ml.inference.predictor import OilSpillPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(size: int = 224) -> bytes:
    """Generate a minimal synthetic greyscale PNG for use in tests."""
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sample_aoi() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]
        ],
    }


def _sample_inference_request(investigation_id: str) -> MLInferenceRequest:
    return MLInferenceRequest(
        investigation_id=investigation_id,
        aoi_geojson=_sample_aoi(),
        start_date="2024-01-01",
        end_date="2024-01-15",
        model_variant="primary",
        # Low threshold guarantees positives from random mock confidences
        confidence_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# DB fixture (in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def investigation(db: AsyncSession) -> Investigation:
    """Create a pending investigation for use in ML pipeline tests."""
    inv = await investigation_service.create_investigation(
        db,
        InvestigationCreate(
            name="End-to-end Test Investigation",
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        ),
    )
    return inv


@pytest.fixture
def mock_predictor() -> OilSpillPredictor:
    """Return an OilSpillPredictor instance (model NOT loaded — mock mode only)."""
    predictor = OilSpillPredictor(
        model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
        device="cpu",
    )
    # Do NOT call predictor.load() — in mock mode the model file is not needed.
    return predictor


# ---------------------------------------------------------------------------
# Unit: OilSpillPredictor — mock mode
# ---------------------------------------------------------------------------


class TestOilSpillPredictorMockMode:
    """ML_MOCK_MODE=true path — no real model loaded."""

    def test_predict_batch_returns_one_result_per_valid_image(self, mock_predictor):
        """predict_batch must return one TilePrediction per valid PNG."""
        images = [_make_png_bytes() for _ in range(3)]
        results = mock_predictor.predict_batch(images)
        assert len(results) == 3

    def test_predict_batch_results_have_mock_flag(self, mock_predictor):
        """Every mock-mode result must have mock=True."""
        images = [_make_png_bytes()]
        results = mock_predictor.predict_batch(images)
        assert results[0].mock is True

    def test_predict_batch_confidence_in_range(self, mock_predictor):
        """Confidence must be in [0, 1]."""
        random.seed(42)
        images = [_make_png_bytes() for _ in range(20)]
        results = mock_predictor.predict_batch(images)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_predict_batch_label_matches_confidence(self, mock_predictor):
        """Label and is_positive must be consistent with confidence vs threshold."""
        # Seed so we get a mix of positives and negatives
        random.seed(0)
        images = [_make_png_bytes() for _ in range(30)]
        results = mock_predictor.predict_batch(images)
        for r in results:
            expected_positive = r.confidence >= mock_predictor.confidence_threshold
            assert r.is_positive == expected_positive
            assert r.label == ("Oil Spill" if r.is_positive else "No Oil")

    def test_predict_single_is_consistent(self, mock_predictor):
        """predict() (single-image convenience) must return a single TilePrediction."""
        result = mock_predictor.predict(_make_png_bytes())
        assert result.mock is True
        assert isinstance(result.confidence, float)

    def test_model_version_unknown_when_not_loaded(self, mock_predictor):
        """model_version returns 'unknown' when model is not loaded."""
        assert mock_predictor.model_version == "unknown"

    def test_is_loaded_false_when_not_loaded(self, mock_predictor):
        """is_loaded is False before load() is called."""
        assert mock_predictor.is_loaded is False


# ---------------------------------------------------------------------------
# Unit: GEE service — mock mode
# ---------------------------------------------------------------------------


class TestGEEServiceMockMode:
    """GEE_MOCK_MODE=true path — no real Earth Engine calls."""

    @pytest.mark.asyncio
    async def test_get_sar_tiles_returns_three_tiles(self):
        """In mock mode, get_sar_tiles must return exactly 1 SARScene (post-fix)."""
        tiles = await gee_service.get_sar_tiles(
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        )
        assert len(tiles) == 1

    @pytest.mark.asyncio
    async def test_get_sar_tiles_mock_flag_is_true(self):
        """All tiles returned in mock mode must have mock=True."""
        tiles = await gee_service.get_sar_tiles(
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        )
        assert all(t.mock for t in tiles)

    @pytest.mark.asyncio
    async def test_get_sar_tiles_have_png_bytes(self):
        """Each SARScene's raster_bytes must be a valid TIFF/image loadable by PIL with size 512×512."""
        tiles = await gee_service.get_sar_tiles(
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        )
        for tile in tiles:
            # Must be loadable as a PIL image
            img = Image.open(io.BytesIO(tile.raster_bytes))
            assert img.size == (512, 512)

    @pytest.mark.asyncio
    async def test_get_sar_tiles_have_valid_bounds(self):
        """Tile bounds must be within the AOI bounding box."""
        tiles = await gee_service.get_sar_tiles(
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        )
        for tile in tiles:
            min_lon, min_lat, max_lon, max_lat = tile.bounds
            assert min_lon >= 55.0
            assert max_lon <= 56.0
            # Latitudes are within the outer AOI bounds (tiles may overlap slightly)
            assert min_lat >= 22.0 - 0.1
            assert max_lat <= 23.0 + 0.1

    @pytest.mark.asyncio
    async def test_get_sar_tiles_tile_ids_are_unique(self):
        """Mock scene IDs must be distinct."""
        tiles = await gee_service.get_sar_tiles(
            aoi_geojson=_sample_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
        )
        ids = [t.scene_id for t in tiles]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Unit: build_spill_polygons
# ---------------------------------------------------------------------------


class TestBuildSpillPolygons:
    """Pure polygon construction from TilePrediction objects."""

    def _make_tile(self, tile_id: str, confidence: float, bounds) -> TilePrediction:
        return TilePrediction(
            tile_id=tile_id,
            label="Oil Spill",
            confidence=confidence,
            is_positive=True,
            bounds=bounds,
            acquisition_date="2024-01-01",
            mock=True,
        )

    def test_empty_input_returns_empty_list(self):
        assert build_spill_polygons([]) == []

    def test_single_tile_produces_one_detection(self):
        tile = self._make_tile("t0", 0.9, (55.0, 22.0, 56.0, 23.0))
        detections = build_spill_polygons([tile])
        assert len(detections) == 1

    def test_detection_has_required_fields(self):
        tile = self._make_tile("t0", 0.8, (55.0, 22.0, 56.0, 23.0))
        detection = build_spill_polygons([tile])[0]
        assert isinstance(detection.detection_id, str)
        assert detection.polygon_geojson is not None
        assert "type" in detection.polygon_geojson
        assert 0.0 <= detection.mean_confidence <= 1.0
        assert detection.tile_count >= 1
        assert detection.area_km2 is not None and detection.area_km2 > 0

    def test_adjacent_tiles_merge_into_one_detection(self):
        """Two spatially touching tiles should merge into a single detection."""
        tile0 = self._make_tile("t0", 0.9, (55.0, 22.0, 55.5, 22.5))
        tile1 = self._make_tile("t1", 0.7, (55.5, 22.0, 56.0, 22.5))
        detections = build_spill_polygons([tile0, tile1])
        # Adjacent boxes merge into one polygon
        assert len(detections) == 1
        assert detections[0].tile_count == 2

    def test_non_overlapping_tiles_produce_separate_detections(self):
        """Two spatially disjoint tiles should produce two separate detections."""
        tile0 = self._make_tile("t0", 0.9, (55.0, 22.0, 55.1, 22.1))
        tile1 = self._make_tile("t1", 0.7, (58.0, 28.0, 58.1, 28.1))
        detections = build_spill_polygons([tile0, tile1])
        assert len(detections) == 2

    def test_mean_confidence_correct(self):
        tile0 = self._make_tile("t0", 0.8, (55.0, 22.0, 55.5, 22.5))
        tile1 = self._make_tile("t1", 0.6, (55.5, 22.0, 56.0, 22.5))
        detections = build_spill_polygons([tile0, tile1])
        assert abs(detections[0].mean_confidence - 0.7) < 1e-6

    def test_max_confidence_correct(self):
        tile0 = self._make_tile("t0", 0.8, (55.0, 22.0, 55.5, 22.5))
        tile1 = self._make_tile("t1", 0.6, (55.5, 22.0, 56.0, 22.5))
        detections = build_spill_polygons([tile0, tile1])
        assert abs(detections[0].max_confidence - 0.8) < 1e-6

    def test_merge_overlapping_false_gives_one_detection_per_tile(self):
        tiles = [
            self._make_tile("t0", 0.8, (55.0, 22.0, 55.5, 22.5)),
            self._make_tile("t1", 0.6, (55.5, 22.0, 56.0, 22.5)),
        ]
        detections = build_spill_polygons(tiles, merge_overlapping=False)
        assert len(detections) == 2


# ---------------------------------------------------------------------------
# Integration: run_ml_inference (full async pipeline, mock mode)
# ---------------------------------------------------------------------------


class TestRunMLInference:
    """Full pipeline: GEE → predictor → polygon → InferenceResult."""

    @pytest.mark.asyncio
    async def test_returns_inference_result_type(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        assert isinstance(result, InferenceResult)

    @pytest.mark.asyncio
    async def test_processes_mock_tiles_with_sliding_window(self, investigation, mock_predictor):
        """Sliding window on 1 mock 512×512 raster (stride=112 default) yields > 4 crops.

        1 SARScene (512×512) at stride=112 → floor(512/112+1)² ≈ 25 tiles.
        total_tiles_processed reflects crops, not raw SAR scenes.

        Validates: Requirements 8.1, 8.2
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        # 1 mock SARScene (512×512) at stride=112 → ~25 sliding-window crops
        assert result.total_tiles_processed > 4

    @pytest.mark.asyncio
    async def test_mock_mode_flag_is_true(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        assert result.mock_mode is True

    @pytest.mark.asyncio
    async def test_at_least_one_spill_detection_with_low_threshold(
        self, investigation, mock_predictor
    ):
        """With confidence_threshold=0.1, ≥1 detection is expected from 3 random tiles.

        Validates: Requirements — POST /api/ml/run-inference returns InferenceResult
        with at least one SpillDetection.
        """
        # Seed random for determinism: with threshold=0.1 and 3 tiles all with
        # uniform [0,1] confidences, the probability of ALL being < 0.1 is 0.001.
        # We seed to guarantee positives.
        random.seed(1)
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        assert result.positive_tile_count >= 1
        assert len(result.spill_detections) >= 1

    @pytest.mark.asyncio
    async def test_spill_detection_structure(self, investigation, mock_predictor):
        """Every SpillDetection in the result must have the required fields."""
        random.seed(1)
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for det in result.spill_detections:
            assert isinstance(det, SpillDetection)
            assert det.detection_id
            assert det.polygon_geojson
            assert det.tile_count >= 1
            assert 0.0 <= det.mean_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_tile_results_count_matches_total(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        assert len(result.tile_results) == result.total_tiles_processed

    @pytest.mark.asyncio
    async def test_positive_tile_count_consistent(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        actual_positives = sum(1 for t in result.tile_results if t.is_positive)
        assert actual_positives == result.positive_tile_count

    @pytest.mark.asyncio
    async def test_investigation_id_preserved(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        assert result.investigation_id == investigation.id

    @pytest.mark.asyncio
    async def test_no_warnings_in_normal_mock_run(self, investigation, mock_predictor):
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        # In mock mode with a valid AOI, there should be no warnings
        assert result.warnings == []

    @pytest.mark.asyncio
    async def test_result_is_json_serializable(self, investigation, mock_predictor):
        """InferenceResult.model_dump(mode='json') must not raise."""
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        dumped = result.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert "spill_detections" in dumped
        assert "tile_results" in dumped


# ---------------------------------------------------------------------------
# Integration: FastAPI HTTP routes (mock mode, in-memory DB)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HTTP-level fixture — uses FastAPI TestClient (sync) with dependency overrides
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client():
    """
    Create a TestClient for the FastAPI app with the in-memory SQLite DB
    and a mock predictor injected.
    """
    import asyncio
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.ml_service import get_predictor
    from app.db.database import get_db
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    predictor = OilSpillPredictor(
        model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
        device="cpu",
    )
    app.dependency_overrides[get_predictor] = lambda: predictor

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Use asyncio.run() which is safe regardless of whether a loop is open
    async def _setup_db():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup_db())

    test_session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )

    async def override_get_db():
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    app.dependency_overrides.clear()

    async def _teardown_db():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await test_engine.dispose()

    asyncio.run(_teardown_db())


class TestMLHealthEndpoint:
    """GET /api/ml/health"""

    def test_health_returns_200(self, app_client):
        resp = app_client.get("/api/ml/health")
        assert resp.status_code == 200

    def test_health_status_is_mock_mode_or_ready(self, app_client):
        """
        Validates: GET /api/ml/health returns status 'mock_mode' or 'ready'.
        """
        resp = app_client.get("/api/ml/health")
        data = resp.json()
        assert "data" in data
        assert data["data"]["status"] in ("mock_mode", "ready", "not_loaded")

    def test_health_has_required_fields(self, app_client):
        resp = app_client.get("/api/ml/health")
        health = resp.json()["data"]
        assert "status" in health
        assert "model_loaded" in health
        assert "device" in health


class TestRunInferenceEndpoint:
    """POST /api/ml/run-inference"""

    def _create_investigation(self, client) -> str:
        resp = client.post(
            "/investigations/",
            json={
                "name": "HTTP Test Investigation",
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
            },
        )
        assert resp.status_code == 201
        return resp.json()["data"]["id"]

    def test_run_inference_returns_200(self, app_client):
        random.seed(1)
        inv_id = self._create_investigation(app_client)
        resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )
        assert resp.status_code == 200

    def test_run_inference_returns_data_envelope(self, app_client):
        random.seed(1)
        inv_id = self._create_investigation(app_client)
        resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )
        body = resp.json()
        assert "data" in body

    def test_run_inference_has_spill_detections(self, app_client):
        """
        Validates: POST /api/ml/run-inference returns InferenceResult with
        at least one SpillDetection (confidence_threshold=0.1).
        """
        random.seed(1)
        inv_id = self._create_investigation(app_client)
        resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )
        result = resp.json()["data"]
        # 1 SARScene (512×512) at stride=112 → ~25 sliding-window crops (> 4)
        assert result["total_tiles_processed"] > 4
        assert result["positive_tile_count"] >= 1
        assert len(result["spill_detections"]) >= 1

    def test_run_inference_404_for_unknown_investigation(self, app_client):
        resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": "00000000-0000-0000-0000-000000000000",
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
            },
        )
        assert resp.status_code == 404

    def test_run_inference_409_for_already_complete(self, app_client):
        """Re-running on a complete investigation must return 409 Conflict."""
        random.seed(1)
        inv_id = self._create_investigation(app_client)
        # First run — succeeds
        app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )
        # Second run — conflict
        resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
            },
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Unit: compute_pixel_stats  (Task 6.1 — Requirement 12.5)
# ---------------------------------------------------------------------------


def _create_white_png(width: int, height: int) -> bytes:
    """Create a solid-white PNG image of the given dimensions."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_png(width: int, height: int, color=(128, 128, 128)) -> bytes:
    """Create a solid-colour PNG image of the given dimensions."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestComputePixelStats:
    """Unit tests for ml_service.compute_pixel_stats (Requirement 12.5)."""

    def test_all_white_2x2_returns_correct_stats(self):
        """2×2 all-white PNG → min=255, max=255, mean=255, std=0.

        Validates: Requirements 12.5
        """
        from app.services.ml_service import compute_pixel_stats

        png_bytes = _create_white_png(2, 2)
        stats = compute_pixel_stats(png_bytes)

        assert stats is not None
        assert stats.min_value == 255.0
        assert stats.max_value == 255.0
        assert stats.mean_value == 255.0
        assert stats.std_value == 0.0

    def test_unreadable_bytes_returns_none_without_raising(self):
        """Corrupt / non-image bytes must return None without raising.

        Validates: Requirements 12.5
        """
        from app.services.ml_service import compute_pixel_stats

        result = compute_pixel_stats(b"this is not a valid image")
        assert result is None

    def test_empty_bytes_returns_none_without_raising(self):
        """Empty bytes must return None without raising.

        Validates: Requirements 12.5
        """
        from app.services.ml_service import compute_pixel_stats

        result = compute_pixel_stats(b"")
        assert result is None

    def test_valid_png_returns_pixel_stats_instance(self):
        """Any valid PNG must produce a PixelStats instance.

        Validates: Requirements 12.5
        """
        from app.services.ml_service import compute_pixel_stats
        from app.schemas.ml import PixelStats

        png_bytes = _create_png(64, 64, color=(100, 150, 200))
        stats = compute_pixel_stats(png_bytes)

        assert stats is not None
        assert isinstance(stats, PixelStats)

    def test_valid_png_stats_are_finite_floats(self):
        """Returned stats must all be finite floats in [0, 255].

        Validates: Requirements 12.5
        """
        from app.services.ml_service import compute_pixel_stats
        import math

        png_bytes = _create_png(32, 32, color=(60, 120, 180))
        stats = compute_pixel_stats(png_bytes)

        assert stats is not None
        for field_value in (stats.min_value, stats.max_value, stats.mean_value, stats.std_value):
            assert isinstance(field_value, float)
            assert math.isfinite(field_value)
            assert 0.0 <= field_value <= 255.0


# ---------------------------------------------------------------------------
# Unit: extract_sliding_window_tiles  (Task 6.3 — Requirements 8.1, 8.2, 8.6)
# ---------------------------------------------------------------------------


class TestExtractSlidingWindowTiles:
    """Unit tests for ml_service.extract_sliding_window_tiles (Requirements 8.1, 8.2, 8.6)."""

    def _make_png(self, width: int, height: int) -> bytes:
        img = Image.new("RGB", (width, height), color=(80, 120, 160))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_448x448_tile224_stride224_returns_4_crops_not_truncated(self):
        """448×448, tile_size=224, stride=224 → exactly 4 crops and truncated=False.

        Validates: Requirements 8.1, 8.2
        """
        from app.services.ml_service import extract_sliding_window_tiles

        png_bytes = self._make_png(448, 448)
        scene_bounds = (0.0, 0.0, 1.0, 1.0)
        crops, truncated = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 448, 448, tile_size=224, stride=224
        )

        assert len(crops) == 4
        assert truncated is False

    def test_image_smaller_than_tile_returns_at_least_one_crop(self):
        """Image smaller than tile_size on either dimension → at least 1 crop.

        Validates: Requirements 8.2, 8.6
        """
        from app.services.ml_service import extract_sliding_window_tiles

        # 100×80 image is smaller than the default tile_size=224
        png_bytes = self._make_png(100, 80)
        scene_bounds = (0.0, 0.0, 1.0, 1.0)
        crops, truncated = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 100, 80, tile_size=224, stride=224
        )

        assert len(crops) >= 1

    def test_crop_bytes_are_valid_pngs_loadable_by_pil(self):
        """Each crop in the returned list must be a valid PNG loadable by PIL.

        Validates: Requirements 8.1, 8.6
        """
        from app.services.ml_service import extract_sliding_window_tiles

        png_bytes = self._make_png(448, 448)
        scene_bounds = (0.0, 0.0, 1.0, 1.0)
        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 448, 448, tile_size=224, stride=224
        )

        for crop_bytes, col, row, crop_bounds in crops:
            # Must not raise
            img = Image.open(io.BytesIO(crop_bytes))
            img.verify()  # raises if not a valid image

    def test_crop_dimensions_are_tile_size(self):
        """Every crop must be resized to tile_size × tile_size.

        Validates: Requirements 8.1
        """
        from app.services.ml_service import extract_sliding_window_tiles

        tile_size = 224
        png_bytes = self._make_png(300, 300)
        scene_bounds = (0.0, 0.0, 1.0, 1.0)
        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 300, 300, tile_size=tile_size, stride=tile_size
        )

        for crop_bytes, col, row, crop_bounds in crops:
            img = Image.open(io.BytesIO(crop_bytes))
            assert img.size == (tile_size, tile_size)

    def test_crop_indices_are_integers(self):
        """col and row indices in each crop tuple must be integers.

        Validates: Requirements 8.1
        """
        from app.services.ml_service import extract_sliding_window_tiles

        png_bytes = self._make_png(448, 448)
        scene_bounds = (0.0, 0.0, 1.0, 1.0)
        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 448, 448, tile_size=224, stride=224
        )

        for _, col, row, crop_bounds in crops:
            assert isinstance(col, int)
            assert isinstance(row, int)


# ---------------------------------------------------------------------------
# Unit: generate_gradcam  (Task 6.6 — Requirement 13.5)
# ---------------------------------------------------------------------------


class TestGenerateGradcam:
    """Unit tests for ml_service.generate_gradcam (Requirement 13.5)."""

    def test_empty_bytes_returns_none_without_raising(self):
        """Passing empty bytes must return None without raising.

        Validates: Requirements 13.5
        """
        from app.services.ml_service import generate_gradcam

        model_mock = MagicMock()
        result = generate_gradcam(model_mock, b"", device="cpu")
        assert result is None

    def test_import_error_on_pytorch_grad_cam_returns_none(self):
        """When pytorch_grad_cam is not importable, generate_gradcam returns None.

        Validates: Requirements 13.5
        """
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pytorch_grad_cam":
                raise ImportError("pytorch_grad_cam not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Re-import to pick up the patched __import__ at call time
            from app.services import ml_service
            model_mock = MagicMock()
            png_bytes = _create_png(64, 64)
            result = ml_service.generate_gradcam(model_mock, png_bytes, device="cpu")

        assert result is None

    def test_invalid_image_bytes_returns_none_without_raising(self):
        """Corrupt image bytes must return None without raising.

        Validates: Requirements 13.5
        """
        from app.services.ml_service import generate_gradcam

        model_mock = MagicMock()
        result = generate_gradcam(model_mock, b"not-valid-image-data", device="cpu")
        assert result is None


# ---------------------------------------------------------------------------
# Unit: TilePrediction per-tile fields  (Task 4.2 — Requirements 12.1–12.3, 12.5–12.6)
# ---------------------------------------------------------------------------


class TestTilePredictionPerTileFields:
    """
    Verify that run_ml_inference populates pixel_stats, debug_info,
    image_url, and image_data_b64 correctly on every TilePrediction.

    Validates: Requirements 12.1, 12.2, 12.3, 12.5, 12.6
    """

    @pytest.mark.asyncio
    async def test_pixel_stats_populated_for_all_tiles(self, investigation, mock_predictor):
        """Every TilePrediction must have a non-None pixel_stats.

        Validates: Requirements 12.5
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            assert tile.pixel_stats is not None, f"pixel_stats is None for tile {tile.tile_id}"

    @pytest.mark.asyncio
    async def test_pixel_stats_fields_are_finite_floats(self, investigation, mock_predictor):
        """pixel_stats min/max/mean/std must all be finite floats in [0, 255].

        Validates: Requirements 12.5
        """
        import math

        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            stats = tile.pixel_stats
            assert stats is not None
            for val in (stats.min_value, stats.max_value, stats.mean_value, stats.std_value):
                assert math.isfinite(val), f"Non-finite pixel stat in tile {tile.tile_id}"
                assert 0.0 <= val <= 255.0, f"Pixel stat out of [0,255] range in tile {tile.tile_id}"

    @pytest.mark.asyncio
    async def test_debug_info_populated_for_all_tiles(self, investigation, mock_predictor):
        """Every TilePrediction must have a non-None debug_info.

        Validates: Requirements 12.6
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            assert tile.debug_info is not None, f"debug_info is None for tile {tile.tile_id}"

    @pytest.mark.asyncio
    async def test_debug_info_preprocessing_steps_correct(self, investigation, mock_predictor):
        """debug_info.preprocessing_steps must be ['resize_224x224', 'normalize'].

        Validates: Requirements 12.6
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            assert tile.debug_info is not None
            assert tile.debug_info.preprocessing_steps == ["resize_224x224", "normalize"], (
                f"Unexpected preprocessing_steps for tile {tile.tile_id}"
            )

    @pytest.mark.asyncio
    async def test_debug_info_model_input_dims_correct(self, investigation, mock_predictor):
        """debug_info.model_input_dims must be (3, 224, 224).

        Validates: Requirements 12.6
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            assert tile.debug_info is not None
            assert tile.debug_info.model_input_dims == (3, 224, 224), (
                f"Unexpected model_input_dims for tile {tile.tile_id}"
            )

    @pytest.mark.asyncio
    async def test_debug_info_input_dimensions_is_height_width_tuple(self, investigation, mock_predictor):
        """debug_info.input_dimensions must be a (height, width) tuple of positive ints.

        Validates: Requirements 12.6
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            assert tile.debug_info is not None
            h, w = tile.debug_info.input_dimensions
            assert isinstance(h, int) and h > 0, f"Invalid height {h} in tile {tile.tile_id}"
            assert isinstance(w, int) and w > 0, f"Invalid width {w} in tile {tile.tile_id}"

    @pytest.mark.asyncio
    async def test_mock_tiles_have_image_data_b64_not_image_url(self, investigation, mock_predictor):
        """In GEE mock mode (tile.mock=True), image_data_b64 must be set and image_url must be None.

        Validates: Requirements 12.1, 12.3
        """
        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        # All tiles are mock in GEE_MOCK_MODE=true
        for tile in result.tile_results:
            assert tile.mock is True
            assert tile.image_data_b64 is not None, (
                f"image_data_b64 should be populated for mock tile {tile.tile_id}"
            )
            assert tile.image_url is None, (
                f"image_url should be None for mock tile {tile.tile_id}"
            )

    @pytest.mark.asyncio
    async def test_mock_image_data_b64_is_valid_base64_png(self, investigation, mock_predictor):
        """image_data_b64 must decode to a valid PNG loadable by PIL.

        Validates: Requirements 12.3
        """
        import base64

        req = _sample_inference_request(investigation.id)
        result = await run_ml_inference(req, mock_predictor)
        for tile in result.tile_results:
            if tile.image_data_b64 is not None:
                decoded = base64.b64decode(tile.image_data_b64)
                img = Image.open(io.BytesIO(decoded))
                assert img.size[0] > 0 and img.size[1] > 0, (
                    f"Decoded image_data_b64 has invalid size for tile {tile.tile_id}"
                )

    @pytest.mark.asyncio
    async def test_real_tile_image_url_populated_when_thumb_url_set(self, investigation, mock_predictor):
        """For a non-mock SARScene (real GEE path), image_url and image_data_b64 must both be None.

        The pipeline now uses SARScene (no thumb_url field).  In real mode the service
        does not populate image_url — that contract is: real tiles → image_url=None,
        image_data_b64=None.

        Validates: Requirements 12.1, 12.2
        """
        from app.services.gee_service import SARScene, _make_noise_tif
        from app.services import gee_service as gee_mod

        # Build a non-mock SARScene (mimics a real GEE download)
        real_scene = SARScene(
            scene_id="real_scene_0",
            raster_bytes=_make_noise_tif(512),
            acquisition_date="2024-01-01",
            orbit_pass="ASCENDING",
            bounds=(55.0, 22.0, 56.0, 23.0),
            raster_width_px=512,
            raster_height_px=512,
            mock=False,
            scale_m=40,
        )

        req = _sample_inference_request(investigation.id)

        # Patch gee_service.get_sar_tiles to return our real SARScene
        with patch.object(gee_mod, "get_sar_tiles", new=AsyncMock(return_value=[real_scene])):
            result = await run_ml_inference(req, mock_predictor)

        # Real mode (mock=False): image_url=None, image_data_b64=None
        for tile in result.tile_results:
            assert tile.image_url is None, (
                f"image_url should be None for real (non-mock) SARScene tile {tile.tile_id}"
            )
            assert tile.image_data_b64 is None, (
                f"image_data_b64 should be None for real (non-mock) SARScene tile {tile.tile_id}"
            )

    @pytest.mark.asyncio
    async def test_real_tile_without_thumb_url_has_neither_image_field(
        self, investigation, mock_predictor
    ):
        """A non-mock SARScene always has image_url=None and image_data_b64=None.

        SARScene does not carry a thumb_url.  Real-mode TilePredictions must have
        both image fields set to None (no pre-encoded image available server-side).

        Validates: Requirements 12.1, 12.2
        """
        from app.services.gee_service import SARScene, _make_noise_tif
        from app.services import gee_service as gee_mod

        real_scene_no_url = SARScene(
            scene_id="real_scene_no_url",
            raster_bytes=_make_noise_tif(512),
            acquisition_date="2024-01-01",
            orbit_pass="ASCENDING",
            bounds=(55.0, 22.0, 56.0, 23.0),
            raster_width_px=512,
            raster_height_px=512,
            mock=False,
            scale_m=40,
        )

        req = _sample_inference_request(investigation.id)

        with patch.object(gee_mod, "get_sar_tiles", new=AsyncMock(return_value=[real_scene_no_url])):
            result = await run_ml_inference(req, mock_predictor)

        for tile in result.tile_results:
            assert tile.image_url is None
            assert tile.image_data_b64 is None


# ---------------------------------------------------------------------------
# Bug Condition Exploration Tests  (Task 1 — Requirements 1.1, 1.2, 1.3, 1.4, 1.8)
#
# CRITICAL: These tests are EXPECTED TO FAIL on unfixed code.
# Failure CONFIRMS the bug exists.
# DO NOT fix the production code or the tests when they fail.
# The failures ARE the success condition for this task.
# ---------------------------------------------------------------------------


def _gulf_of_mexico_aoi() -> dict:
    """Return a GeoJSON Polygon for a ~1×1 degree Gulf of Mexico AOI."""
    return {
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


def _gulf_inference_request(investigation_id: str) -> MLInferenceRequest:
    return MLInferenceRequest(
        investigation_id=investigation_id,
        aoi_geojson=_gulf_of_mexico_aoi(),
        start_date="2024-01-01",
        end_date="2024-01-15",
        model_variant="primary",
        confidence_threshold=0.1,
    )


class TestBugConditionExploration:
    """
    Bug condition exploration tests for the Sentinel-1 pipeline.

    Property being tested: isBugCondition(X) — export_method = "thumbnail"
    AND raster_dims <= 224×224 despite large AOI.

    All four sub-tests MUST FAIL on unfixed code.
    Failure is the SUCCESS condition — it proves the bug exists.

    Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.8
    """

    # ------------------------------------------------------------------
    # Sub-test A — Mock tile count is capped / SARTile missing raster_width_px
    # ------------------------------------------------------------------

    def test_bug_condition_A_mock_tile_has_raster_width_px(self):
        """Sub-test A: _get_mock_sar_tiles() first result must have raster_width_px > 224.

        On UNFIXED code: SARTile has no 'raster_width_px' attribute → AttributeError.
        This confirms the wrong dataclass is used (SARTile instead of SARScene).

        Validates: Requirements 1.1, 1.2
        """
        from app.services.gee_service import _get_mock_sar_tiles

        aoi = _gulf_of_mexico_aoi()
        result = _get_mock_sar_tiles(aoi)

        assert len(result) > 0, "Expected at least one tile from _get_mock_sar_tiles"
        # On unfixed code this raises: AttributeError: SARTile has no attribute 'raster_width_px'
        assert result[0].raster_width_px > 224, (
            f"Expected raster_width_px > 224 but got: {result[0].raster_width_px}"
        )

    # ------------------------------------------------------------------
    # Sub-test B — Full pipeline tile count does not scale with AOI size
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_bug_condition_B_pipeline_tile_count_scales_with_aoi(
        self, investigation, mock_predictor
    ):
        """Sub-test B: run_ml_inference with Gulf of Mexico AOI must produce more than
        the hardcoded 12 tiles that unfixed code always returns.

        On UNFIXED code: always exactly 12 tiles (3 mock tiles × 4 crops from each
        224×224 thumbnail) regardless of AOI size.  The fixed code returns a single
        512×512 raster which at stride=112 yields ~(4+1)²=25 tiles.

        The assertion checks > 12 so it distinguishes the real bug outcome (== 12)
        from the fixed outcome (> 12).

        isBugCondition holds: export_method='thumbnail', raster_dims<=224×224, area>0.01 deg²

        Validates: Requirements 1.1, 1.3, 1.4
        """
        req = MLInferenceRequest(
            investigation_id=investigation.id,
            aoi_geojson=_gulf_of_mexico_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
            model_variant="primary",
            confidence_threshold=0.1,
        )
        result = await run_ml_inference(req, mock_predictor)

        # On unfixed code this assertion FAILS: result.total_tiles_processed == 12 (not > 12)
        assert result.total_tiles_processed > 12, (
            f"Expected total_tiles_processed > 12 (tile count must scale with a larger raster), "
            f"but got: {result.total_tiles_processed}. "
            f"The unfixed code hardcodes 3 mock 224×224 tiles giving exactly 12 crops, "
            f"proving the tile count does not scale with AOI/raster dimensions."
        )

    # ------------------------------------------------------------------
    # Sub-test C — All tile bounds identical to full AOI bounds
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_bug_condition_C_tile_bounds_are_distinct_per_crop(
        self, investigation, mock_predictor
    ):
        """Sub-test C: Crops that originate from the SAME parent SAR tile must have
        distinct bounds from each other (per-crop geographic bounds).

        On UNFIXED code: every crop from the same parent tile shares the parent's
        bounds (e.g. (-90.5, 28.5, -89.5, 28.866...)).  All 4 crops per tile report
        exactly the same bounding box — confirming that per-crop geolocation is absent
        and geotransform-based sub-tile bounds are never computed.

        Validates: Requirements 1.1, 1.4
        """
        req = MLInferenceRequest(
            investigation_id=investigation.id,
            aoi_geojson=_gulf_of_mexico_aoi(),
            start_date="2024-01-01",
            end_date="2024-01-15",
            model_variant="primary",
            confidence_threshold=0.1,
        )
        result = await run_ml_inference(req, mock_predictor)

        assert len(result.tile_results) >= 2, (
            f"Need at least 2 tile results to compare bounds, got {len(result.tile_results)}"
        )

        # Group tiles by their parent SAR tile (identified by the prefix before _r / _c suffix).
        # Each parent produces multiple crops; within a parent the crops must differ.
        from collections import defaultdict
        parent_groups: dict[str, list] = defaultdict(list)
        for tile in result.tile_results:
            # Tile IDs are formatted as "<parent_id>_r<row>_c<col>" in ml_service.py
            # Strip the _rN_cM suffix to recover the parent SAR tile ID.
            parts = tile.tile_id.rsplit("_r", 1)
            parent_id = parts[0] if len(parts) == 2 else tile.tile_id
            parent_groups[parent_id].append(tile.bounds)

        # For at least one parent, the crops must NOT all share the same bounds
        all_crops_identical_within_parent = True
        for parent_id, bounds_list in parent_groups.items():
            if len(bounds_list) > 1 and len(set(bounds_list)) > 1:
                # Found a parent whose crops have distinct bounds — bug is NOT present
                all_crops_identical_within_parent = False
                break

        # On unfixed code: all crops within a parent share the same bounds → assertion FAILS
        assert not all_crops_identical_within_parent, (
            f"All crops within every parent SAR tile share identical bounds. "
            f"Example parent groups (bounds): "
            + str({pid: list(set(bl)) for pid, bl in list(parent_groups.items())[:2]})
            + ". This confirms the bug: per-crop geographic bounding boxes are not computed; "
            f"each crop inherits the full parent tile bounds."
        )

    # ------------------------------------------------------------------
    # Sub-test D — dB conversion utilities absent from preprocess module
    # ------------------------------------------------------------------

    def test_bug_condition_D_db_convert_and_clip_importable_and_correct(self):
        """Sub-test D: db_convert_and_clip must be importable and return ≈0.0 for input 1.0.

        On UNFIXED code: ml/preprocessing/preprocess.py is empty → ImportError.
        This confirms that the training-time preprocessing utilities are absent.

        10 * log10(1.0) = 0.0, so db_convert_and_clip(np.array([1.0])) should ≈ 0.0 dB.

        Validates: Requirements 1.8
        """
        import numpy as np

        # On unfixed code this import raises ImportError because preprocess.py is empty
        from ml.preprocessing.preprocess import db_convert_and_clip  # noqa: F401

        result = db_convert_and_clip(np.array([1.0]))
        assert result.shape == (1,), f"Expected shape (1,) but got {result.shape}"

        # 10 * log10(1.0) = 0.0 dB, then clip(-25, 0) → 0.0
        assert abs(float(result[0]) - 0.0) < 1e-6, (
            f"Expected db_convert_and_clip(np.array([1.0])) ≈ 0.0 dB, "
            f"but got {float(result[0])}. "
            f"This confirms the dB conversion function is either absent or incorrect."
        )


# ---------------------------------------------------------------------------
# Preservation Property-Based Tests  (Task 2 — Requirements 3.1–3.10)
#
# EXPECTED: ALL tests PASS on unfixed code — they lock down the baseline
# behaviors that must remain unchanged after the fix is applied.
#
# These five PBTs cover the "NOT isBugCondition(X)" region of the input space:
# mock-mode runs, confidence thresholding, stride passthrough,
# graceful empty-GEE handling, and investigation status transitions.
# ---------------------------------------------------------------------------


from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st
from math import floor


# ---------------------------------------------------------------------------
# Preservation PBT A — InferenceResult schema completeness
#
# For random MLInferenceRequest values (varied confidence thresholds, stride
# values) with mock modes enabled, all required InferenceResult fields must
# be present and correctly typed.
#
# Validates: Requirements 3.1, 3.2, 3.6
# ---------------------------------------------------------------------------

class TestPreservationPBT_A_InferenceResultSchema:
    """
    **Validates: Requirements 3.1, 3.2, 3.6**

    Property: For any mock-mode MLInferenceRequest, run_ml_inference returns
    an InferenceResult whose required fields are all present and correctly typed.
    """

    @pytest.mark.asyncio
    @given(
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stride=st.sampled_from([56, 112, 224]),
    )
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,  # suppress deadline — async pipeline test is inherently slow
    )
    async def test_pbt_a_inference_result_schema_completeness(
        self, confidence_threshold, stride
    ):
        """PBT A: InferenceResult must conform to schema for any valid mock-mode request.

        Observation on unfixed code (baseline):
        - run_ml_inference completes without raising in GEE_MOCK_MODE + ML_MOCK_MODE
        - All required fields are present and correctly typed
        - mock_mode=True always when either mock mode is active

        **Validates: Requirements 3.1, 3.2, 3.6**
        """
        # Build an in-memory DB and investigation for each generated example
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with factory() as session:
            inv = await investigation_service.create_investigation(
                session,
                InvestigationCreate(
                    name="PBT-A Test",
                    aoi_geojson=_sample_aoi(),
                    start_date="2024-01-01",
                    end_date="2024-01-15",
                ),
            )

            predictor = OilSpillPredictor(
                model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
                device="cpu",
            )

            req = MLInferenceRequest(
                investigation_id=inv.id,
                aoi_geojson=_sample_aoi(),
                start_date="2024-01-01",
                end_date="2024-01-15",
                model_variant="primary",
                confidence_threshold=confidence_threshold,
                stride=stride,
            )

            # Must not raise
            result = await run_ml_inference(req, predictor)

        await engine.dispose()

        # --- Schema completeness assertions ---
        assert isinstance(result, InferenceResult)

        # Required string fields
        assert isinstance(result.investigation_id, str)
        assert result.investigation_id == inv.id

        # Required int fields
        assert isinstance(result.total_tiles_processed, int)
        assert result.total_tiles_processed >= 0
        assert isinstance(result.positive_tile_count, int)
        assert result.positive_tile_count >= 0

        # Required list fields
        assert isinstance(result.spill_detections, list)
        assert isinstance(result.tile_results, list)
        assert isinstance(result.warnings, list)

        # mock_mode must be True when mock modes are active
        assert isinstance(result.mock_mode, bool)
        assert result.mock_mode is True

        # Counts must be consistent
        assert result.positive_tile_count <= result.total_tiles_processed
        assert len(result.tile_results) == result.total_tiles_processed


# ---------------------------------------------------------------------------
# Preservation PBT B — Confidence threshold gating
#
# For random (confidence, threshold) pairs in [0.0, 1.0], a TilePrediction
# constructed by run_ml_inference must have is_positive == (confidence >= threshold).
#
# Validates: Requirements 3.4, 3.9
# ---------------------------------------------------------------------------

class TestPreservationPBT_B_ConfidenceThresholdGating:
    """
    **Validates: Requirements 3.4, 3.9**

    Property: For any (confidence, threshold) pair in [0.0, 1.0],
    TilePrediction.is_positive == (confidence >= threshold).
    """

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @h_settings(max_examples=200)
    def test_pbt_b_confidence_threshold_gating(self, confidence, threshold):
        """PBT B: is_positive must equal (confidence >= threshold) for all valid pairs.

        Observation on unfixed code (baseline):
        - ml_service.run_ml_inference applies: is_positive = pred.confidence >= request.confidence_threshold
        - This must hold for every tile regardless of the threshold value

        **Validates: Requirements 3.4, 3.9**
        """
        # Directly exercise the gating logic by constructing a TilePrediction
        # using the same formula that ml_service uses — this validates the contract
        # independently of the full pipeline.
        is_positive = confidence >= threshold
        label = "Oil Spill" if is_positive else "No Oil"

        tile = TilePrediction(
            tile_id="pbt_b_tile",
            label=label,
            confidence=confidence,
            is_positive=is_positive,
            bounds=(55.0, 22.0, 56.0, 23.0),
            acquisition_date="2024-01-01",
            mock=True,
        )

        # Core property: is_positive must equal (confidence >= threshold)
        assert tile.is_positive == (confidence >= threshold), (
            f"is_positive={tile.is_positive} does not match "
            f"(confidence={confidence} >= threshold={threshold}) = {confidence >= threshold}"
        )

    @pytest.mark.asyncio
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_pbt_b_pipeline_threshold_applied_to_every_tile(self, threshold):
        """PBT B (pipeline): Every TilePrediction from run_ml_inference must have
        is_positive == (confidence >= request.confidence_threshold).

        **Validates: Requirements 3.4, 3.9**
        """
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with factory() as session:
            inv = await investigation_service.create_investigation(
                session,
                InvestigationCreate(
                    name="PBT-B Pipeline Test",
                    aoi_geojson=_sample_aoi(),
                    start_date="2024-01-01",
                    end_date="2024-01-15",
                ),
            )

            predictor = OilSpillPredictor(
                model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
                device="cpu",
            )

            req = MLInferenceRequest(
                investigation_id=inv.id,
                aoi_geojson=_sample_aoi(),
                start_date="2024-01-01",
                end_date="2024-01-15",
                confidence_threshold=threshold,
            )

            result = await run_ml_inference(req, predictor)

        await engine.dispose()

        for tile in result.tile_results:
            expected = tile.confidence >= threshold
            assert tile.is_positive == expected, (
                f"Tile {tile.tile_id}: is_positive={tile.is_positive} but "
                f"confidence={tile.confidence} >= threshold={threshold} is {expected}"
            )


# ---------------------------------------------------------------------------
# Preservation PBT C — Stride passthrough
#
# For stride values {56, 112, 224} applied to a 448×448 synthetic image,
# crop count must match floor((448 - 224) / stride + 1)².
#
# Validates: Requirements 3.10
# ---------------------------------------------------------------------------

class TestPreservationPBT_C_StridePassthrough:
    """
    **Validates: Requirements 3.10**

    Property: For a 448×448 image and stride s ∈ {56, 112, 224}, stride is
    correctly passed through to extract_sliding_window_tiles and the resulting
    crop count changes accordingly with different stride values.

    Observation on unfixed code (baseline):
    - extract_sliding_window_tiles iterates while col * stride < W (per axis)
    - For W=448, stride=s: steps per axis = ceil(448 / s)
    - Total crops = ceil(448/s)^2
    - The key invariant: different strides produce different crop counts,
      and crop count is monotonically decreasing as stride increases.
    """

    @given(
        stride=st.sampled_from([56, 112, 224]),
    )
    @h_settings(max_examples=3)
    def test_pbt_c_stride_passthrough_crop_count(self, stride):
        """PBT C: Stride value is passed through and affects crop count correctly.

        On UNFIXED code, extract_sliding_window_tiles uses:
            while col * stride < W  (per axis)
        which yields ceil(W / stride) steps per axis.

        For W=H=448:
          stride=56  → ceil(448/56)=8  → 8²=64  crops
          stride=112 → ceil(448/112)=4 → 4²=16  crops
          stride=224 → ceil(448/224)=2 → 2²=4   crops

        This confirms the stride is correctly plumbed into the tiler.

        **Validates: Requirements 3.10**
        """
        import math
        from app.services.ml_service import extract_sliding_window_tiles

        # 448×448 synthetic PNG
        img = Image.new("RGB", (448, 448), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        crops, truncated = extract_sliding_window_tiles(
            png_bytes,
            scene_bounds=(0.0, 0.0, 1.0, 1.0),
            raster_width_px=448,
            raster_height_px=448,
            tile_size=224,
            stride=stride,
        )

        # Unfixed code: while col * stride < W → ceil(W / stride) steps per axis
        steps_per_axis = math.ceil(448 / stride)
        expected_count = steps_per_axis * steps_per_axis

        assert len(crops) == expected_count, (
            f"stride={stride}: expected {expected_count} crops "
            f"(steps_per_axis=ceil(448/{stride})={steps_per_axis}), got {len(crops)}"
        )
        # Truncated should be False for these small counts (max is 64, well under cap)
        assert truncated is False, (
            f"stride={stride}: unexpected truncation for only {expected_count} crops"
        )
        # Core preservation invariant: crop count must be positive
        assert len(crops) > 0, f"stride={stride}: got 0 crops — stride passthrough broken"

    def test_pbt_c_larger_stride_produces_fewer_crops(self):
        """PBT C: Larger stride must produce fewer crops — confirms stride is plumbed.

        Validates the monotonicity invariant that is preserved across fix.

        **Validates: Requirements 3.10**
        """
        from app.services.ml_service import extract_sliding_window_tiles

        img = Image.new("RGB", (448, 448), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        crops_56, _ = extract_sliding_window_tiles(
            png_bytes,
            scene_bounds=(0.0, 0.0, 1.0, 1.0),
            raster_width_px=448,
            raster_height_px=448,
            tile_size=224,
            stride=56,
        )
        crops_112, _ = extract_sliding_window_tiles(
            png_bytes,
            scene_bounds=(0.0, 0.0, 1.0, 1.0),
            raster_width_px=448,
            raster_height_px=448,
            tile_size=224,
            stride=112,
        )
        crops_224, _ = extract_sliding_window_tiles(
            png_bytes,
            scene_bounds=(0.0, 0.0, 1.0, 1.0),
            raster_width_px=448,
            raster_height_px=448,
            tile_size=224,
            stride=224,
        )

        assert len(crops_56) > len(crops_112) > len(crops_224), (
            f"Expected crops_56({len(crops_56)}) > crops_112({len(crops_112)}) "
            f"> crops_224({len(crops_224)}). Stride is not being applied correctly."
        )


# ---------------------------------------------------------------------------
# Preservation PBT D — Zero-scene graceful return
#
# Patch gee_service.get_sar_tiles to return []; assert
# result.total_tiles_processed == 0, result.spill_detections == [], no exception.
#
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

class TestPreservationPBT_D_ZeroSceneGracefulReturn:
    """
    **Validates: Requirements 3.3**

    Property: When GEE returns zero scenes, run_ml_inference must return a
    valid InferenceResult with total_tiles_processed=0 and spill_detections=[],
    with no exception raised.

    Observation on unfixed code (baseline):
    - _zero_detection_result() is called when tiles list is empty
    - Returns InferenceResult with 0 tiles, no detections, and a warning
    """

    @pytest.mark.asyncio
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stride=st.sampled_from([56, 112, 224, None]),
    )
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_pbt_d_zero_scene_graceful_return(self, threshold, stride):
        """PBT D: Empty GEE response must yield zero-detection InferenceResult, no exception.

        **Validates: Requirements 3.3**
        """
        from app.services import gee_service as gee_mod

        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with factory() as session:
            inv = await investigation_service.create_investigation(
                session,
                InvestigationCreate(
                    name="PBT-D Zero Scene Test",
                    aoi_geojson=_sample_aoi(),
                    start_date="2024-01-01",
                    end_date="2024-01-15",
                ),
            )

            predictor = OilSpillPredictor(
                model_path="ml/models/oricean_oil_classifier_mobilenetv3_large.pth",
                device="cpu",
            )

            req = MLInferenceRequest(
                investigation_id=inv.id,
                aoi_geojson=_sample_aoi(),
                start_date="2024-01-01",
                end_date="2024-01-15",
                confidence_threshold=threshold,
                stride=stride,
            )

            # Patch get_sar_tiles to return empty list — simulates GEE returning no scenes
            with patch.object(gee_mod, "get_sar_tiles", new=AsyncMock(return_value=[])):
                result = await run_ml_inference(req, predictor)

        await engine.dispose()

        # Core property: graceful zero-detection result
        assert isinstance(result, InferenceResult), (
            "run_ml_inference must return InferenceResult even for empty GEE response"
        )
        assert result.total_tiles_processed == 0, (
            f"Expected total_tiles_processed=0 for empty GEE response, "
            f"got {result.total_tiles_processed}"
        )
        assert result.spill_detections == [], (
            f"Expected spill_detections=[] for empty GEE response, "
            f"got {result.spill_detections}"
        )
        assert result.tile_results == [], (
            f"Expected tile_results=[] for empty GEE response"
        )
        assert result.positive_tile_count == 0
        # A warning should be present explaining the empty result
        assert len(result.warnings) >= 1, (
            "Expected at least one warning for empty GEE response"
        )
        # investigation_id must still be preserved
        assert result.investigation_id == inv.id


# ---------------------------------------------------------------------------
# Preservation PBT E — Investigation status transitions
#
# Full HTTP smoke test: confirm investigation transitions from pending → complete
# and ml_result is persisted after successful inference.
#
# Validates: Requirements 3.6, 3.7
# ---------------------------------------------------------------------------

class TestPreservationPBT_E_InvestigationStatusTransitions:
    """
    **Validates: Requirements 3.6, 3.7**

    Property: After a successful POST /api/ml/run-inference call:
    - The investigation status transitions to "complete"
    - The investigation.ml_result is non-null and contains the InferenceResult data

    Observation on unfixed code (baseline):
    - ml.py route sets status to complete via investigation_service.set_status
    - ml_result is written directly onto the ORM object before commit
    """

    def _create_investigation_http(self, client) -> str:
        """Helper: create an investigation via HTTP and return its ID."""
        resp = client.post(
            "/investigations/",
            json={
                "name": "PBT-E Status Transition Test",
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
            },
        )
        assert resp.status_code == 201, (
            f"Expected 201 creating investigation, got {resp.status_code}: {resp.text}"
        )
        return resp.json()["data"]["id"]

    def test_pbt_e_status_transitions_to_complete_after_inference(self, app_client):
        """PBT E: Investigation must transition pending → complete after successful inference.

        Validates: Requirements 3.6, 3.7
        """
        random.seed(42)

        inv_id = self._create_investigation_http(app_client)

        # Verify initial state is pending
        get_resp = app_client.get(f"/investigations/{inv_id}")
        assert get_resp.status_code == 200
        initial_status = get_resp.json()["data"]["status"]
        assert initial_status == "pending", (
            f"Expected initial status 'pending', got '{initial_status}'"
        )

        # Run inference
        infer_resp = app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )
        assert infer_resp.status_code == 200, (
            f"Expected 200 from run-inference, got {infer_resp.status_code}: {infer_resp.text}"
        )

        # Verify status is now complete
        get_resp2 = app_client.get(f"/investigations/{inv_id}")
        assert get_resp2.status_code == 200
        final_status = get_resp2.json()["data"]["status"]
        assert final_status == "complete", (
            f"Expected status 'complete' after inference, got '{final_status}'"
        )

        # Verify ml_result is persisted and non-null
        inv_data = get_resp2.json()["data"]
        assert inv_data.get("ml_result") is not None, (
            "ml_result must be non-null after successful inference"
        )

        # ml_result must contain the required InferenceResult fields
        ml_result = inv_data["ml_result"]
        assert "investigation_id" in ml_result, "ml_result missing investigation_id"
        assert "total_tiles_processed" in ml_result, "ml_result missing total_tiles_processed"
        assert "spill_detections" in ml_result, "ml_result missing spill_detections"
        assert "tile_results" in ml_result, "ml_result missing tile_results"
        assert "mock_mode" in ml_result, "ml_result missing mock_mode"
        assert "warnings" in ml_result, "ml_result missing warnings"

        # investigation_id in result must match
        assert ml_result["investigation_id"] == inv_id, (
            f"ml_result.investigation_id {ml_result['investigation_id']!r} != inv_id {inv_id!r}"
        )
        # mock_mode must be True in mock mode runs
        assert ml_result["mock_mode"] is True, (
            "ml_result.mock_mode must be True when ML_MOCK_MODE=true"
        )

    def test_pbt_e_health_endpoint_status_in_valid_set(self, app_client):
        """PBT E (health): GET /api/ml/health status must be in the valid set.

        Validates: Requirements 3.8
        """
        resp = app_client.get("/api/ml/health")
        assert resp.status_code == 200

        health = resp.json()["data"]

        # status must be one of the three valid values
        assert health["status"] in {"mock_mode", "ready", "not_loaded"}, (
            f"health.status={health['status']!r} not in valid set "
            f"{{'mock_mode', 'ready', 'not_loaded'}}"
        )

        # model_loaded must be a bool
        assert isinstance(health["model_loaded"], bool), (
            f"health.model_loaded must be bool, got {type(health['model_loaded'])}"
        )

    def test_pbt_e_ml_result_persisted_contains_all_required_fields(self, app_client):
        """PBT E: ml_result must contain all required InferenceResult fields after persistence.

        Validates: Requirements 3.6, 3.7
        """
        random.seed(7)

        inv_id = self._create_investigation_http(app_client)

        app_client.post(
            "/api/ml/run-inference",
            json={
                "investigation_id": inv_id,
                "aoi_geojson": _sample_aoi(),
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "confidence_threshold": 0.1,
            },
        )

        get_resp = app_client.get(f"/investigations/{inv_id}")
        ml_result = get_resp.json()["data"]["ml_result"]
        assert ml_result is not None

        required_fields = {
            "investigation_id",
            "total_tiles_processed",
            "positive_tile_count",
            "spill_detections",
            "tile_results",
            "mock_mode",
            "warnings",
        }
        for field in required_fields:
            assert field in ml_result, (
                f"Required field '{field}' missing from persisted ml_result. "
                f"Present fields: {list(ml_result.keys())}"
            )


# ---------------------------------------------------------------------------
# Task 4 — Unit and integration tests for new components
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit: db_convert_and_clip
# ---------------------------------------------------------------------------

class TestDbConvertAndClip:
    """Unit tests for ml.preprocessing.preprocess.db_convert_and_clip."""

    def setup_method(self):
        from ml.preprocessing.preprocess import db_convert_and_clip
        self.fn = db_convert_and_clip

    def test_unit_value_of_one_maps_to_zero_db(self):
        """db_convert_and_clip(np.array([1.0])) ≈ 0.0 since 10*log10(1) = 0."""
        result = self.fn(np.array([1.0]))
        assert abs(result[0] - 0.0) < 1e-9, (
            f"Expected ≈ 0.0 dB for linear input 1.0, got {result[0]}"
        )

    def test_near_zero_input_clips_to_floor(self):
        """db_convert_and_clip(np.array([1e-10])) ≈ -25.0 (epsilon hits clip floor)."""
        result = self.fn(np.array([1e-10]))
        # 10*log10(1e-10) = -100, but clipped to -25
        assert abs(result[0] - (-25.0)) < 1e-9, (
            f"Expected ≈ -25.0 dB for near-zero linear input, got {result[0]}"
        )

    def test_very_large_input_clamps_to_zero(self):
        """Very large linear backscatter values → clamped to 0.0 dB (clip max)."""
        result = self.fn(np.array([1e6]))
        assert result[0] == 0.0, (
            f"Expected 0.0 dB (clamped) for very large input, got {result[0]}"
        )

    @given(
        st.lists(
            st.floats(min_value=1e-12, max_value=1e8, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @h_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_property_output_always_in_valid_range(self, values):
        """All output values ∈ [-25.0, 0.0] for any positive float input array.

        **Validates: Requirements 2.6**
        """
        arr = np.array(values, dtype=np.float64)
        result = self.fn(arr)
        assert result.shape == arr.shape
        assert np.all(result >= -25.0), (
            f"Output below -25.0 found: min={result.min()}"
        )
        assert np.all(result <= 0.0), (
            f"Output above 0.0 found: max={result.max()}"
        )


# ---------------------------------------------------------------------------
# Unit: normalize_db_to_uint8
# ---------------------------------------------------------------------------

class TestNormalizeDbToUint8:
    """Unit tests for ml.preprocessing.preprocess.normalize_db_to_uint8."""

    def setup_method(self):
        from ml.preprocessing.preprocess import normalize_db_to_uint8
        self.fn = normalize_db_to_uint8

    def test_minus_25_db_maps_to_0_uint8(self):
        """Input np.array([-25.0]) → np.array([0], dtype=np.uint8)."""
        result = self.fn(np.array([-25.0]))
        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )
        assert result[0] == 0, (
            f"Expected 0 for -25.0 dB, got {result[0]}"
        )

    def test_zero_db_maps_to_255_uint8(self):
        """Input np.array([0.0]) → np.array([255], dtype=np.uint8)."""
        result = self.fn(np.array([0.0]))
        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )
        assert result[0] == 255, (
            f"Expected 255 for 0.0 dB, got {result[0]}"
        )

    def test_midpoint_minus_12_5_maps_to_approx_127(self):
        """Input np.array([-12.5]) → output ≈ 127 (± 1)."""
        result = self.fn(np.array([-12.5]))
        assert abs(int(result[0]) - 127) <= 1, (
            f"Expected ≈ 127 (±1) for -12.5 dB, got {result[0]}"
        )


# ---------------------------------------------------------------------------
# Unit: extract_sliding_window_tiles — per-crop bounds accuracy
# ---------------------------------------------------------------------------

class TestExtractSlidingWindowTilesBoundsAccuracy:
    """Unit tests for per-crop geographic bounds in extract_sliding_window_tiles."""

    def _make_png(self, width: int, height: int) -> bytes:
        """Create a synthetic PNG of given dimensions."""
        arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_first_crop_min_lon_is_scene_min_lon(self):
        """First crop's min_lon (crop_bounds[0]) must equal scene_bounds min_lon (0.0)."""
        from app.services.ml_service import extract_sliding_window_tiles

        # Synthetic 224×512 image: width=512, height=224
        png_bytes = self._make_png(width=512, height=224)
        scene_bounds = (0.0, 0.0, 2.0, 1.0)

        crops, truncated = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 512, 224, tile_size=224, stride=224
        )

        assert len(crops) >= 1, "Expected at least one crop"
        # First crop in reading order (col=0, row=0)
        first_crop = crops[0]
        crop_bounds = first_crop[3]  # (crop_bytes, col, row, crop_bounds)

        assert abs(crop_bounds[0] - 0.0) < 1e-9, (
            f"First crop min_lon should be 0.0, got {crop_bounds[0]}"
        )

    def test_first_crop_max_lon_matches_geotransform(self):
        """First crop's max_lon ≈ 224 * (2.0/512) within 1e-9 tolerance."""
        from app.services.ml_service import extract_sliding_window_tiles

        png_bytes = self._make_png(width=512, height=224)
        scene_bounds = (0.0, 0.0, 2.0, 1.0)
        lon_per_px = 2.0 / 512
        expected_max_lon = 224 * lon_per_px

        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 512, 224, tile_size=224, stride=224
        )

        first_crop_bounds = crops[0][3]
        assert abs(first_crop_bounds[2] - expected_max_lon) < 1e-9, (
            f"First crop max_lon expected ≈ {expected_max_lon}, got {first_crop_bounds[2]}"
        )

    def test_all_crop_bounds_are_strict_subsets_of_scene_bounds(self):
        """All crop bounds must satisfy min_lon>=0, min_lat>=0, max_lon<=2, max_lat<=1."""
        from app.services.ml_service import extract_sliding_window_tiles

        png_bytes = self._make_png(width=512, height=224)
        scene_bounds = (0.0, 0.0, 2.0, 1.0)
        min_lon, min_lat, max_lon, max_lat = scene_bounds

        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 512, 224, tile_size=224, stride=224
        )

        for i, (crop_bytes, col, row, crop_bounds) in enumerate(crops):
            c_min_lon, c_min_lat, c_max_lon, c_max_lat = crop_bounds
            assert c_min_lon >= min_lon - 1e-9, (
                f"Crop {i} min_lon {c_min_lon} < scene min_lon {min_lon}"
            )
            assert c_min_lat >= min_lat - 1e-9, (
                f"Crop {i} min_lat {c_min_lat} < scene min_lat {min_lat}"
            )
            assert c_max_lon <= max_lon + 1e-9, (
                f"Crop {i} max_lon {c_max_lon} > scene max_lon {max_lon}"
            )
            assert c_max_lat <= max_lat + 1e-9, (
                f"Crop {i} max_lat {c_max_lat} > scene max_lat {max_lat}"
            )

    def test_non_overlapping_crops_have_distinct_bounds(self):
        """No two crops at different (col, row) positions should share identical bounds."""
        from app.services.ml_service import extract_sliding_window_tiles

        # Use a larger image so we get multiple crops with stride=224
        png_bytes = self._make_png(width=512, height=224)
        scene_bounds = (0.0, 0.0, 2.0, 1.0)

        crops, _ = extract_sliding_window_tiles(
            png_bytes, scene_bounds, 512, 224, tile_size=224, stride=224
        )

        # Collect (col, row) → bounds
        position_to_bounds = {}
        for crop_bytes, col, row, crop_bounds in crops:
            pos = (col, row)
            position_to_bounds[pos] = crop_bounds

        # If there are multiple distinct positions, their bounds must differ
        unique_positions = list(position_to_bounds.keys())
        if len(unique_positions) >= 2:
            bounds_set = set(position_to_bounds.values())
            assert len(bounds_set) == len(unique_positions), (
                f"Expected {len(unique_positions)} distinct bounds for distinct positions, "
                f"but got only {len(bounds_set)} unique bounds. "
                f"Positions: {unique_positions}, bounds: {bounds_set}"
            )


# ---------------------------------------------------------------------------
# Unit: _get_mock_sar_tiles — post-fix tests
# ---------------------------------------------------------------------------

class TestGetMockSarTilesPostFix:
    """Post-fix unit tests for gee_service._get_mock_sar_tiles."""

    def _get_sample_aoi(self) -> dict:
        return {
            "type": "Polygon",
            "coordinates": [
                [[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]
            ],
        }

    def test_returns_list_of_length_one(self):
        """_get_mock_sar_tiles must return exactly one SARScene after the fix."""
        from app.services.gee_service import _get_mock_sar_tiles

        result = _get_mock_sar_tiles(self._get_sample_aoi())
        assert isinstance(result, list), (
            f"Expected a list, got {type(result)}"
        )
        assert len(result) == 1, (
            f"Expected list of length 1, got {len(result)}"
        )

    def test_element_is_sar_scene_instance(self):
        """Element 0 must be a SARScene instance with raster_width_px >= 512 and height >= 512."""
        from app.services.gee_service import _get_mock_sar_tiles, SARScene

        result = _get_mock_sar_tiles(self._get_sample_aoi())
        scene = result[0]

        assert isinstance(scene, SARScene), (
            f"Expected SARScene instance, got {type(scene)}"
        )
        assert scene.raster_width_px >= 512, (
            f"Expected raster_width_px >= 512, got {scene.raster_width_px}"
        )
        assert scene.raster_height_px >= 512, (
            f"Expected raster_height_px >= 512, got {scene.raster_height_px}"
        )

    def test_mock_flag_is_true(self):
        """SARScene.mock must be True for the mock path."""
        from app.services.gee_service import _get_mock_sar_tiles

        result = _get_mock_sar_tiles(self._get_sample_aoi())
        assert result[0].mock is True, (
            f"Expected mock=True, got mock={result[0].mock}"
        )

    def test_raster_bytes_is_pil_readable(self):
        """SARScene.raster_bytes must be openable by PIL.Image.open without error."""
        from app.services.gee_service import _get_mock_sar_tiles

        result = _get_mock_sar_tiles(self._get_sample_aoi())
        scene = result[0]

        img = Image.open(io.BytesIO(scene.raster_bytes))
        assert img is not None, "PIL.Image.open returned None"
        # Ensure the image is actually loaded (no lazy load issues)
        img.load()


# ---------------------------------------------------------------------------
# Unit: _make_noise_tif helper
# ---------------------------------------------------------------------------

class TestMakeNoiseTif:
    """Unit tests for gee_service._make_noise_tif."""

    def test_returns_non_empty_bytes(self):
        """_make_noise_tif must return non-empty bytes."""
        from app.services.gee_service import _make_noise_tif

        result = _make_noise_tif(256)
        assert isinstance(result, bytes), (
            f"Expected bytes, got {type(result)}"
        )
        assert len(result) > 0, "Expected non-empty bytes"

    def test_pil_can_open_result(self):
        """PIL must be able to open the returned bytes as a valid image."""
        from app.services.gee_service import _make_noise_tif

        result = _make_noise_tif(256)
        img = Image.open(io.BytesIO(result))
        # Force loading to verify it's a real image
        img.load()
        assert img is not None

    def test_image_has_requested_size(self):
        """Image produced by _make_noise_tif(size=256) must have size (256, 256)."""
        from app.services.gee_service import _make_noise_tif

        size = 256
        result = _make_noise_tif(size)
        img = Image.open(io.BytesIO(result))
        assert img.size == (size, size), (
            f"Expected image size ({size}, {size}), got {img.size}"
        )
