"""
Focused unit tests for the officer-alert significance rule and the vessel
attribution evidence gates.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
os.environ.setdefault("MODEL_PATH", "/dev/null")
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")

from datetime import datetime, timezone

from app.services import alert_service
from app.services.attribution_service import score_candidates
from app.schemas.attribution import VesselRecord


# ---------------------------------------------------------------------------
# alert_service.evaluate_detection
# ---------------------------------------------------------------------------


def test_evaluate_detection_significant():
    d = alert_service.evaluate_detection(
        {"positive_tile_count": 10, "spill_detections": [{"max_confidence": 0.95, "area_km2": 500}]},
        min_confidence=0.6,
        min_positive_tiles=3,
    )
    assert d.significant is True
    assert d.severity in {"critical", "high"}


def test_evaluate_detection_below_threshold():
    d = alert_service.evaluate_detection(
        {"positive_tile_count": 1, "spill_detections": [{"max_confidence": 0.4}]},
        min_confidence=0.6,
        min_positive_tiles=3,
    )
    assert d.significant is False


def test_evaluate_detection_no_detection():
    d = alert_service.evaluate_detection({"positive_tile_count": 0, "spill_detections": []}, 0.6, 3)
    assert d.significant is False
    assert "no spill" in d.reason


def test_clean_numbers_tolerates_inline_comment():
    nums = alert_service._clean_numbers("+919975091377   # comma-separated E.164 numbers, e.g. +14155550123")
    assert nums == ["+919975091377"]


# ---------------------------------------------------------------------------
# attribution evidence gates
# ---------------------------------------------------------------------------

_REGION = {
    "type": "Polygon",
    "coordinates": [[[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]],
}


def _candidate_dict(vessel, min_dist_km, points_near, rw_start, rw_end):
    return {
        "vessel": vessel,
        "min_distance_km": min_dist_km,
        "entry_time": rw_start.isoformat(),
        "exit_time": rw_end.isoformat(),
        "closest_time": rw_start.isoformat(),
        "temporal_score": 0.5,
        "overlap_minutes": 120.0,
        "max_gap_minutes": 5.0,
        "in_buffer_fraction": 0.5 if min_dist_km < 5 else 0.0,
        "points_near_source": points_near,
        "dwell_hours": 2.0,
        "num_positions": 20,
        "heading": 90.0,
        "angle_diff": 10.0,
        "rw_start": rw_start,
        "rw_end": rw_end,
    }


def test_far_vessel_is_not_qualified_but_still_scored():
    rw_start = datetime(2024, 1, 14, tzinfo=timezone.utc)
    rw_end = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    near = VesselRecord(mmsi="111111111", name="NEAR", track=[(55.5, 22.5, "2024-01-14T06:00:00Z"), (55.6, 22.6, "2024-01-14T12:00:00Z")])
    far = VesselRecord(mmsi="222222222", name="FAR", track=[(70.0, 40.0, "2024-01-14T06:00:00Z"), (70.1, 40.1, "2024-01-14T12:00:00Z")])

    ranked = score_candidates(
        candidates=[
            _candidate_dict(near, 1.0, 5, rw_start, rw_end),
            _candidate_dict(far, 900.0, 0, rw_start, rw_end),
        ],
        source_reconstruction_result=None,
        detection_time=rw_end,
        max_distance_km=12.0,
        max_time_diff_hours=96.0,
        min_trajectory_overlap=0.05,
        min_points_near_source=2,
        min_score=12.0,
    )
    by_mmsi = {c.mmsi: c for c in ranked}
    assert len(ranked) == 2  # every scored vessel is still returned
    assert by_mmsi["111111111"].qualified is True
    assert by_mmsi["111111111"].qualification_reasons  # non-empty
    assert by_mmsi["222222222"].qualified is False
