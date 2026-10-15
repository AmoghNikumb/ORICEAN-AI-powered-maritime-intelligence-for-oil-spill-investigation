"""
Property-based tests for the Source Reconstruction and Vessel Attribution pipelines.

Validates: Requirements 1, 2, 3 (source reconstruction correctness),
           Requirements 4, 5, 6 (attribution scoring and filtering).

Tests use Hypothesis with @given / @settings decorators.

Properties covered:
  Property 1 — Corridor strictly larger than detection region
  Property 2 — score_candidates output sorted descending
  Property 3 — Attribution score in [0.0, 100.0]
  Property 4 — Score breakdown weights sum to 1.0
  Property 5 — Release window satisfies half-lookback formula
  Property 6 — Source confidence always in [0.0, 1.0]
  Property 7 — Invalid drift params raise DriftServiceError
  Property 8 — All filtered candidates satisfy spatial+temporal criteria
  Property 9 — Every candidate has at least 4 evidence strings
"""

import asyncio
import math
import os

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
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from shapely.geometry import Point, shape

# ---------------------------------------------------------------------------
# App imports (safe after env stubs)
# ---------------------------------------------------------------------------
from app.schemas.attribution import AISQueryResult, VesselRecord
from app.services.attribution_service import filter_candidates, score_candidates
from app.services.drift_service import DriftServiceError, compute_source_corridor

# ---------------------------------------------------------------------------
# Shared fixed test data
# ---------------------------------------------------------------------------

_FIXED_DETECTIONS = [
    {
        "polygon_geojson": {
            "type": "Polygon",
            "coordinates": [
                [[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]
            ],
        },
        "mean_confidence": 0.85,
        "area_km2": 10.0,
    }
]

_FIXED_TIMESTAMP = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

_FIXED_REGION_GEOJSON = {
    "type": "Polygon",
    "coordinates": [
        [[55.0, 22.0], [56.0, 22.0], [56.0, 23.0], [55.0, 23.0], [55.0, 22.0]]
    ],
}

_FIXED_RW_START = datetime(2024, 1, 13, 0, 0, 0, tzinfo=timezone.utc)
_FIXED_RW_END = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
_FIXED_BUFFER_KM = 30.0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def valid_detection_strategy(draw):
    """Generate a valid spill detection dict with a small polygon."""
    cx = draw(st.floats(min_value=50.0, max_value=60.0))
    cy = draw(st.floats(min_value=18.0, max_value=26.0))
    size = draw(st.floats(min_value=0.01, max_value=0.5))

    # Filter out non-finite values
    assume(math.isfinite(cx) and math.isfinite(cy) and math.isfinite(size))

    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [cx - size, cy - size],
                [cx + size, cy - size],
                [cx + size, cy + size],
                [cx - size, cy + size],
                [cx - size, cy - size],
            ]
        ],
    }
    confidence = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)))
    if confidence is not None:
        assume(math.isfinite(confidence))
    area = draw(st.floats(min_value=0.1, max_value=100.0))
    assume(math.isfinite(area))
    return {"polygon_geojson": polygon, "mean_confidence": confidence, "area_km2": area}


@st.composite
def candidate_dict_strategy(draw):
    """Generate an intermediate candidate dict matching the shape expected by score_candidates."""
    mmsi = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Nd",)),
            min_size=9,
            max_size=9,
        )
    )

    rw_start = datetime(2024, 1, 13, 0, 0, 0, tzinfo=timezone.utc)
    rw_end = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Generate track with 2-5 points within the window
    n_points = draw(st.integers(min_value=2, max_value=5))
    track = []
    for _ in range(n_points):
        lon = draw(st.floats(min_value=54.0, max_value=57.0))
        lat = draw(st.floats(min_value=21.0, max_value=24.0))
        assume(math.isfinite(lon) and math.isfinite(lat))
        offset_hours = draw(st.floats(min_value=0.0, max_value=60.0))
        assume(math.isfinite(offset_hours))
        ts_dt = rw_start + timedelta(hours=offset_hours)
        ts = ts_dt.isoformat().replace("+00:00", "Z")
        track.append((lon, lat, ts))

    # Sort track by timestamp string (ISO-8601 sorts correctly)
    track.sort(key=lambda x: x[2])

    vessel = VesselRecord(mmsi=mmsi, track=track)

    min_dist = draw(st.floats(min_value=0.0, max_value=19.9))
    assume(math.isfinite(min_dist))
    temporal_score = draw(st.floats(min_value=0.0, max_value=1.0))
    assume(math.isfinite(temporal_score))
    overlap_minutes = draw(st.floats(min_value=0.0, max_value=60.0))
    assume(math.isfinite(overlap_minutes))
    max_gap_minutes = draw(st.floats(min_value=0.0, max_value=120.0))
    assume(math.isfinite(max_gap_minutes))

    heading = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=359.9)))
    if heading is not None:
        assume(math.isfinite(heading))
    angle_diff = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=180.0)))
    if angle_diff is not None:
        assume(math.isfinite(angle_diff))

    return {
        "vessel": vessel,
        "min_distance_km": min_dist,
        "entry_time": track[0][2] if track else None,
        "exit_time": track[-1][2] if track else None,
        "temporal_score": temporal_score,
        "overlap_minutes": overlap_minutes,
        "max_gap_minutes": max_gap_minutes,
        "heading": heading,
        "angle_diff": angle_diff,
        "rw_start": rw_start,
        "rw_end": rw_end,
    }


@st.composite
def vessel_strategy(draw):
    """Generate a VesselRecord with a mix of in-window and out-of-window track points."""
    mmsi = draw(st.text(alphabet="0123456789", min_size=9, max_size=9))
    n_points = draw(st.integers(min_value=1, max_value=5))

    track = []
    for _ in range(n_points):
        lon = draw(st.floats(min_value=54.0, max_value=58.0))
        lat = draw(st.floats(min_value=20.0, max_value=25.0))
        assume(math.isfinite(lon) and math.isfinite(lat))

        # Mix of in-window and out-of-window timestamps
        use_in_window = draw(st.booleans())
        if use_in_window:
            offset_hours = draw(st.floats(min_value=0.0, max_value=60.0))
            assume(math.isfinite(offset_hours))
            ts_dt = _FIXED_RW_START + timedelta(hours=offset_hours)
        else:
            offset_hours = draw(st.floats(min_value=1.0, max_value=100.0))
            assume(math.isfinite(offset_hours))
            ts_dt = _FIXED_RW_END + timedelta(hours=offset_hours)

        ts = ts_dt.isoformat().replace("+00:00", "Z")
        track.append((lon, lat, ts))

    track.sort(key=lambda x: x[2])
    return VesselRecord(mmsi=mmsi, track=track)


# ===========================================================================
# Property 1: Corridor strictly larger than detection region
# Validates: Requirements 1 (source reconstruction expands geometry)
# ===========================================================================


@given(
    detections=st.lists(valid_detection_strategy(), min_size=1, max_size=10),
    drift_speed=st.floats(min_value=0.01, max_value=5.0),
    lookback=st.integers(min_value=1, max_value=336),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_corridor_strictly_larger_than_detection_region(detections, drift_speed, lookback):
    """**Validates: Requirements 1**

    For any valid spill detection list and drift params, the output corridor
    polygon must be strictly larger in both width and height than the bounding
    box of the input detection region.
    """
    assume(math.isfinite(drift_speed))

    output = asyncio.run(
        compute_source_corridor(
            spill_detections=detections,
            acquisition_timestamp=_FIXED_TIMESTAMP,
            drift_speed_m_s=drift_speed,
            lookback_hours=lookback,
        )
    )

    # Parse output corridor
    corridor_geom = shape(output.source_corridor_geojson)
    c_minx, c_miny, c_maxx, c_maxy = corridor_geom.bounds
    corridor_width = c_maxx - c_minx
    corridor_height = c_maxy - c_miny

    # Parse input detection union
    input_geoms = [shape(d["polygon_geojson"]) for d in detections]
    from shapely.ops import unary_union
    input_union = unary_union(input_geoms)
    d_minx, d_miny, d_maxx, d_maxy = input_union.bounds
    detection_width = d_maxx - d_minx
    detection_height = d_maxy - d_miny

    assert corridor_width > detection_width, (
        f"Corridor width {corridor_width} not > detection width {detection_width}"
    )
    assert corridor_height > detection_height, (
        f"Corridor height {corridor_height} not > detection height {detection_height}"
    )


# ===========================================================================
# Property 5: Release window satisfies half-lookback formula
# Validates: Requirements 1 (temporal window computation)
# ===========================================================================


@given(
    t0=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
    lookback_hours=st.integers(min_value=1, max_value=336),
)
@settings(max_examples=100)
def test_release_window_satisfies_half_lookback_formula(t0, lookback_hours):
    """**Validates: Requirements 1**

    The release window start must equal t0 - lookback_hours, and the release
    window end must equal t0 - (lookback_hours * 0.5).
    """
    output = asyncio.run(
        compute_source_corridor(
            spill_detections=_FIXED_DETECTIONS,
            acquisition_timestamp=t0,
            lookback_hours=lookback_hours,
        )
    )

    expected_start = t0 - timedelta(hours=lookback_hours)
    expected_end = t0 - timedelta(hours=lookback_hours * 0.5)

    # Normalise both sides to UTC-aware for comparison
    actual_start = output.release_window_start
    actual_end = output.release_window_end

    if actual_start.tzinfo is None:
        actual_start = actual_start.replace(tzinfo=timezone.utc)
    if actual_end.tzinfo is None:
        actual_end = actual_end.replace(tzinfo=timezone.utc)
    if expected_start.tzinfo is None:
        expected_start = expected_start.replace(tzinfo=timezone.utc)
    if expected_end.tzinfo is None:
        expected_end = expected_end.replace(tzinfo=timezone.utc)

    assert actual_start == expected_start, (
        f"release_window_start {actual_start} != expected {expected_start}"
    )
    assert actual_end == expected_end, (
        f"release_window_end {actual_end} != expected {expected_end}"
    )


# ===========================================================================
# Property 6: Source confidence always in [0.0, 1.0]
# Validates: Requirements 1 (confidence score bounds)
# ===========================================================================


@given(
    drift_speed=st.floats(min_value=0.01, max_value=5.0),
    lookback=st.integers(min_value=1, max_value=336),
)
@settings(max_examples=100)
def test_source_confidence_in_unit_interval(drift_speed, lookback):
    """**Validates: Requirements 1**

    source_confidence must always be in [0.0, 1.0] for any valid input
    combination.
    """
    assume(math.isfinite(drift_speed))

    output = asyncio.run(
        compute_source_corridor(
            spill_detections=_FIXED_DETECTIONS,
            acquisition_timestamp=_FIXED_TIMESTAMP,
            drift_speed_m_s=drift_speed,
            lookback_hours=lookback,
        )
    )

    assert 0.0 <= output.source_confidence <= 1.0, (
        f"source_confidence {output.source_confidence} out of [0.0, 1.0]"
    )


# ===========================================================================
# Property 7: Invalid drift params raise DriftServiceError
# Validates: Requirements 1 (parameter validation)
# ===========================================================================


_invalid_speed_strategy = st.one_of(
    # Below minimum
    st.floats(min_value=-100.0, max_value=0.009, allow_nan=False, allow_infinity=False),
    # Above maximum
    st.floats(min_value=5.001, max_value=100.0, allow_nan=False, allow_infinity=False),
)

_invalid_hours_strategy = st.one_of(
    # Below minimum
    st.integers(min_value=-10000, max_value=0),
    # Above maximum
    st.integers(min_value=337, max_value=10000),
)


@given(
    invalid_params=st.one_of(
        _invalid_speed_strategy.map(lambda s: ("speed", s)),
        _invalid_hours_strategy.map(lambda h: ("hours", h)),
    )
)
@settings(max_examples=100)
def test_invalid_drift_params_raise_error(invalid_params):
    """**Validates: Requirements 1**

    compute_source_corridor must raise DriftServiceError for any drift_speed_m_s
    outside [0.01, 5.0] or lookback_hours outside [1, 336].
    """
    param_type, param_value = invalid_params

    if param_type == "speed":
        speed = param_value
        hours = 48
        # Exclude exact boundary values — strictly invalid
        assume(speed < 0.01 or speed > 5.0)
    else:
        speed = 0.3
        hours = param_value
        assume(hours < 1 or hours > 336)

    with pytest.raises(DriftServiceError):
        asyncio.run(
            compute_source_corridor(
                spill_detections=_FIXED_DETECTIONS,
                acquisition_timestamp=_FIXED_TIMESTAMP,
                drift_speed_m_s=speed,
                lookback_hours=hours,
            )
        )


# ===========================================================================
# Property 2: score_candidates output sorted descending
# Validates: Requirements 2 (candidate ranking order)
# ===========================================================================


@given(candidates=st.lists(candidate_dict_strategy(), min_size=1, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_score_candidates_sorted_descending(candidates):
    """**Validates: Requirements 2**

    score_candidates() must return results monotonically non-increasing in
    attribution_score. For tied scores the output must be ascending by
    min_distance_km.
    """
    result = score_candidates(candidates, None)

    for i in range(len(result) - 1):
        a = result[i]
        b = result[i + 1]
        assert a.attribution_score >= b.attribution_score, (
            f"Result not sorted descending at index {i}: "
            f"{a.attribution_score} < {b.attribution_score}"
        )
        # For tied scores, check ascending min_distance_km
        if a.attribution_score == b.attribution_score:
            assert a.min_distance_km <= b.min_distance_km, (
                f"Tied attribution_score={a.attribution_score} but "
                f"min_distance_km not ascending: {a.min_distance_km} > {b.min_distance_km}"
            )


# ===========================================================================
# Property 3: Attribution score in [0.0, 100.0]
# Validates: Requirements 2 (score bounds)
# ===========================================================================


@given(candidates=st.lists(candidate_dict_strategy(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_attribution_score_in_valid_range(candidates):
    """**Validates: Requirements 2**

    Every RankedCandidate returned by score_candidates must have
    attribution_score in [0.0, 100.0].
    """
    result = score_candidates(candidates, None)

    for c in result:
        assert 0.0 <= c.attribution_score <= 100.0, (
            f"attribution_score {c.attribution_score} out of [0.0, 100.0] "
            f"for vessel {c.mmsi}"
        )


# ===========================================================================
# Property 4: Score breakdown weights sum to 1.0
# Validates: Requirements 2 (weight consistency)
# ===========================================================================


@given(candidates=st.lists(candidate_dict_strategy(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_score_breakdown_weights_sum_to_one(candidates):
    """**Validates: Requirements 2**

    For every RankedCandidate, the weights in score_breakdown must sum to
    1.0 ±0.001.
    """
    result = score_candidates(candidates, None)

    for c in result:
        weight_sum = sum(c.score_breakdown.weights.values())
        assert abs(weight_sum - 1.0) <= 0.001, (
            f"Weights sum {weight_sum} deviates from 1.0 for vessel {c.mmsi}: "
            f"weights={c.score_breakdown.weights}"
        )


# ===========================================================================
# Property 8: All filtered candidates satisfy spatial+temporal criteria
# Validates: Requirements 3 (filter correctness)
# ===========================================================================


@given(vessels=st.lists(vessel_strategy(), min_size=0, max_size=20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_all_filtered_candidates_satisfy_spatial_temporal_criteria(vessels):
    """**Validates: Requirements 3**

    Every vessel returned by filter_candidates must have at least one track
    point that is:
    1. Within the release window [rw_start, rw_end]
    2. Within buffer_km of the region geometry

    Additionally, every vessel from the input that has NO qualifying point
    must NOT appear in the output.
    """
    ais_result = AISQueryResult(vessels=vessels, warnings=[])
    region_geom = shape(_FIXED_REGION_GEOJSON)

    result = filter_candidates(
        ais_result=ais_result,
        region_geojson=_FIXED_REGION_GEOJSON,
        release_window_start=_FIXED_RW_START,
        release_window_end=_FIXED_RW_END,
        buffer_km=_FIXED_BUFFER_KM,
    )

    # Build a set of MMSIs that have at least one qualifying point
    qualifying_mmsis: set[str] = set()
    for vessel in vessels:
        for lon, lat, ts_str in vessel.track:
            try:
                from dateutil import parser as dateutil_parser
                t = dateutil_parser.isoparse(ts_str)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            if not (_FIXED_RW_START <= t <= _FIXED_RW_END):
                continue

            pt = Point(lon, lat)
            dist_km = pt.distance(region_geom) * 111.0
            if dist_km <= _FIXED_BUFFER_KM:
                qualifying_mmsis.add(vessel.mmsi)
                break

    output_mmsis = {cand["vessel"].mmsi for cand in result}

    # All returned candidates must have had a qualifying point
    for cand in result:
        vessel_mmsi = cand["vessel"].mmsi
        assert vessel_mmsi in qualifying_mmsis, (
            f"Vessel {vessel_mmsi} returned by filter_candidates but has no "
            "track point within the release window AND within buffer_km"
        )

    # No vessel without a qualifying point should appear in output
    for vessel in vessels:
        if vessel.mmsi not in qualifying_mmsis:
            assert vessel.mmsi not in output_mmsis, (
                f"Vessel {vessel.mmsi} should have been filtered out but appears in output"
            )


# ===========================================================================
# Property 9: Every candidate has at least 4 evidence strings
# Validates: Requirements 2 (evidence completeness)
# ===========================================================================


@given(candidates=st.lists(candidate_dict_strategy(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_every_candidate_has_four_evidence_strings(candidates):
    """**Validates: Requirements 2**

    Every RankedCandidate returned by score_candidates must have at least
    4 evidence strings — one per scoring dimension.
    """
    result = score_candidates(candidates, None)

    for c in result:
        assert len(c.evidence) >= 4, (
            f"Vessel {c.mmsi} has only {len(c.evidence)} evidence strings, "
            "expected at least 4"
        )
