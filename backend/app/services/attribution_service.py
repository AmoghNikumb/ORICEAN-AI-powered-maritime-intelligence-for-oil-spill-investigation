"""
Attribution service for ORICEAN — candidate filtering, scoring, and ranking.

Responsibilities:
- ``filter_candidates()``  — spatial + temporal filter over an ``AISQueryResult``
  to produce a list of intermediate candidate dicts carrying the raw
  measurements needed for scoring.
- Sub-score helpers: ``_compute_spatial_score`` (proximity),
  ``_compute_temporal_score`` (window coverage), ``_compute_trajectory_score``
  (heading alignment — retained for the evidence note),
  ``_compute_in_buffer_stats`` (trajectory overlap + dwell + closest fix), and
  ``_compute_ais_anomaly_score`` (legacy gap metric, still computed).
- ``score_candidates()`` — consumes the candidate dicts, computes the four
  weighted sub-scores (spatial proximity, temporal consistency, trajectory
  overlap, time spent near region), assembles ``ScoreBreakdown``, generates 4
  evidence strings, sorts descending by ``attribution_score`` (ties broken
  ascending by ``min_distance_km`` then MMSI), assigns 1-based ranks, and
  returns a list of ``RankedCandidate`` Pydantic models. The ranking expresses
  consistency with the spill, not causation.

Intermediate candidate dict shape (produced by ``filter_candidates``,
consumed by ``score_candidates``):
    {
        "vessel":          VesselRecord,
        "min_distance_km": float,
        "entry_time":      str | None,   # ISO-8601 UTC of first point inside buffer
        "exit_time":       str | None,   # ISO-8601 UTC of last point inside buffer
        "temporal_score":  float,        # pre-computed temporal overlap ratio [0, 1]
        "overlap_minutes": float,        # raw overlap duration in minutes
        "max_gap_minutes": float,        # largest AIS gap inside release window
        "heading":         float | None, # estimated heading in degrees (or None)
        "angle_diff":      float | None, # angle vs default drift vector (or None)
        "rw_start":        datetime,     # release window start (UTC-aware)
        "rw_end":          datetime,     # release window end (UTC-aware)
    }

Track point format: (longitude, latitude, timestamp_utc_string)
Shapely distance: returns degrees → multiply by 111.0 to get km.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dateutil_parser
from shapely.geometry import Point, shape

from app.schemas.attribution import AISQueryResult, RankedCandidate, ScoreBreakdown, VesselRecord

logger = logging.getLogger(__name__)

# Approximate km per degree at the equator — used for Shapely distance conversion.
_KM_PER_DEG: float = 111.0

# Default drift vector bearing when source reconstruction data is unavailable.
_DEFAULT_DRIFT_BEARING: float = 0.0


# ---------------------------------------------------------------------------
# Internal timestamp helpers
# ---------------------------------------------------------------------------


def _parse_track_time(ts: str) -> datetime:
    """Parse an ISO-8601 UTC track-point timestamp to an aware datetime.

    Args:
        ts: ISO-8601 UTC string, e.g. ``'2024-01-15T12:00:00Z'``.

    Returns:
        Timezone-aware ``datetime`` in UTC.
    """
    dt = dateutil_parser.isoparse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a timezone-aware UTC datetime.

    Args:
        dt: A ``datetime``; may be naive (assumed UTC) or timezone-aware.

    Returns:
        Timezone-aware ``datetime`` in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Sub-score private helpers
# ---------------------------------------------------------------------------


def _compute_spatial_score(min_distance_km: float, buffer_km: float) -> float:
    """Spatial proximity score in [0.0, 1.0].

    ``1.0`` when the vessel is at or inside the region boundary (distance = 0);
    decreases linearly to ``0.0`` at ``buffer_km``.

    Args:
        min_distance_km: Minimum distance in km between any track point and
            the source corridor geometry.
        buffer_km: Spatial search buffer radius in km.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if buffer_km <= 0.0:
        return 1.0 if min_distance_km == 0.0 else 0.0
    return max(0.0, 1.0 - (min_distance_km / buffer_km))


def _compute_temporal_score(
    track: list[tuple[float, float, str]],
    release_window_start: datetime,
    release_window_end: datetime,
) -> tuple[float, float]:
    """Temporal overlap score and raw overlap duration in minutes.

    Treats consecutive track points as interval endpoints and clips each
    interval to ``[release_window_start, release_window_end]``.

    Args:
        track: List of ``(lon, lat, timestamp_utc)`` tuples ordered
            chronologically ascending.
        release_window_start: Start of the release window (UTC-aware).
        release_window_end: End of the release window (UTC-aware).

    Returns:
        Tuple ``(temporal_score, overlap_minutes)``.  ``temporal_score`` is
        in ``[0.0, 1.0]``; ``overlap_minutes`` is the raw overlap duration.
    """
    rw_start = _ensure_utc(release_window_start)
    rw_end = _ensure_utc(release_window_end)
    window_seconds = (rw_end - rw_start).total_seconds()

    if window_seconds <= 0.0:
        return 0.0, 0.0

    overlap_seconds = 0.0
    for i in range(len(track) - 1):
        try:
            t_a = _parse_track_time(track[i][2])
            t_b = _parse_track_time(track[i + 1][2])
        except Exception:
            continue
        clip_start = max(t_a, rw_start)
        clip_end = min(t_b, rw_end)
        if clip_end > clip_start:
            overlap_seconds += (clip_end - clip_start).total_seconds()

    overlap_minutes = overlap_seconds / 60.0
    temporal_score = min(1.0, overlap_seconds / window_seconds)
    return temporal_score, overlap_minutes


def _compute_trajectory_score(
    track: list[tuple[float, float, str]],
    drift_bearing: float,
) -> tuple[float, float | None, float | None]:
    """Heading-alignment trajectory score.

    Estimates the vessel's heading from the bearing between the first two
    distinct lat/lon positions in the track (ORICEAN track points are
    ``(lon, lat, ts_utc)`` with no explicit heading field).

    Score mapping:
        - ``angle_diff ≤ 30°``  → ``1.0``
        - ``30° < angle_diff ≤ 60°`` → ``0.5``
        - ``angle_diff > 60°``  → ``0.0``
        - No usable heading → ``0.0``

    Args:
        track: List of ``(lon, lat, timestamp_utc)`` tuples.
        drift_bearing: Reference bearing of the drift vector in degrees.

    Returns:
        Tuple ``(trajectory_score, heading, angle_diff)``.  ``heading`` and
        ``angle_diff`` are ``None`` when no heading could be derived.
    """
    heading: float | None = None

    for i in range(len(track) - 1):
        lon1, lat1 = track[i][0], track[i][1]
        lon2, lat2 = track[i + 1][0], track[i + 1][1]
        if lon1 == lon2 and lat1 == lat2:
            continue
        # Rhumb-line bearing (approximate — adequate for short segments).
        d_lon = math.radians(lon2 - lon1)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        x = math.sin(d_lon) * math.cos(lat2_r)
        y = (
            math.cos(lat1_r) * math.sin(lat2_r)
            - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
        )
        heading = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
        break

    if heading is None:
        return 0.0, None, None

    angle_diff = abs((heading - drift_bearing + 180.0) % 360.0 - 180.0)

    if angle_diff <= 30.0:
        score = 1.0
    elif angle_diff <= 60.0:
        score = 0.5
    else:
        score = 0.0

    return score, heading, angle_diff


def _compute_ais_anomaly_score(
    track: list[tuple[float, float, str]],
    release_window_start: datetime,
    release_window_end: datetime,
) -> tuple[float, float]:
    """AIS transmission-gap anomaly score.

    Collects all track-point timestamps inside the release window and
    measures the largest gap between consecutive entries.

    Score mapping:
        - ``max_gap > 60 min``   → ``1.0``
        - ``15 ≤ max_gap ≤ 60 min`` → ``0.5``
        - ``max_gap < 15 min`` (or < 2 in-window points) → ``0.0``

    Args:
        track: List of ``(lon, lat, timestamp_utc)`` tuples.
        release_window_start: Start of the release window (UTC-aware).
        release_window_end: End of the release window (UTC-aware).

    Returns:
        Tuple ``(ais_anomaly_score, max_gap_minutes)``.
    """
    rw_start = _ensure_utc(release_window_start)
    rw_end = _ensure_utc(release_window_end)

    times_in_window: list[datetime] = []
    for point in track:
        try:
            t = _parse_track_time(point[2])
        except Exception:
            continue
        if rw_start <= t <= rw_end:
            times_in_window.append(t)

    if len(times_in_window) < 2:
        return 0.0, 0.0

    times_in_window.sort()
    max_gap_seconds = max(
        (times_in_window[i + 1] - times_in_window[i]).total_seconds()
        for i in range(len(times_in_window) - 1)
    )
    max_gap_minutes = max_gap_seconds / 60.0

    if max_gap_minutes > 60.0:
        score = 1.0
    elif max_gap_minutes >= 15.0:
        score = 0.5
    else:
        score = 0.0

    return score, max_gap_minutes


def _compute_in_buffer_stats(
    track: list[tuple[float, float, str]],
    region_geom,
    buffer_km: float,
    rw_start: datetime,
    rw_end: datetime,
) -> tuple[float, str | None, float, int, int]:
    """Trajectory-overlap and dwell statistics for a vessel track vs the region.

    Walks the (chronologically sorted) track and, for the points that fall
    inside the release window, measures:

    - ``in_buffer_fraction`` — share of in-window points within ``buffer_km``
      of ``region_geom`` (the "trajectory overlap" signal, in ``[0, 1]``).
    - ``closest_time`` — ISO-8601 timestamp of the point nearest the region.
    - ``dwell_hours`` — summed duration of consecutive point-to-point segments
      whose *midpoint* sits inside the buffer, clipped to the release window.
    - ``in_window_count`` — number of in-window track points.
    - ``in_buffer_count`` — number of in-window points within ``buffer_km`` of
      the region (the "AIS points near the source" gate signal).

    Args:
        track: ``(lon, lat, ts_utc)`` points, chronologically ascending.
        region_geom: Shapely geometry of the source region.
        buffer_km: Search buffer radius in km.
        rw_start / rw_end: Release-window bounds (UTC-aware).

    Returns:
        ``(in_buffer_fraction, closest_time, dwell_hours, in_window_count,
        in_buffer_count)``.
    """
    rw_start = _ensure_utc(rw_start)
    rw_end = _ensure_utc(rw_end)

    pts: list[tuple[float, float, datetime, float]] = []
    closest_time: str | None = None
    closest_dist = float("inf")

    for lon, lat, ts in track:
        try:
            t = _parse_track_time(ts)
        except Exception:
            continue
        if not (rw_start <= t <= rw_end):
            continue
        dist_km = Point(lon, lat).distance(region_geom) * _KM_PER_DEG
        pts.append((lon, lat, t, dist_km))
        if dist_km < closest_dist:
            closest_dist = dist_km
            closest_time = ts

    if not pts:
        return 0.0, None, 0.0, 0, 0

    in_buffer = sum(1 for *_x, d in pts if d <= buffer_km)
    in_buffer_fraction = in_buffer / len(pts)

    dwell_seconds = 0.0
    for i in range(len(pts) - 1):
        _, _, t_a, d_a = pts[i]
        _, _, t_b, d_b = pts[i + 1]
        if (d_a + d_b) / 2.0 <= buffer_km:
            seg = (min(t_b, rw_end) - max(t_a, rw_start)).total_seconds()
            if seg > 0:
                dwell_seconds += seg

    return in_buffer_fraction, closest_time, dwell_seconds / 3600.0, len(pts), in_buffer


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def filter_candidates(
    ais_result: AISQueryResult,
    region_geojson: dict,
    release_window_start: datetime,
    release_window_end: datetime,
    buffer_km: float = 20.0,
) -> list[dict[str, Any]]:
    """Filter AIS vessels to those passing spatial and temporal criteria.

    A vessel passes when **at least one** of its track points satisfies both:
    1. Timestamp falls within ``[release_window_start, release_window_end]``.
    2. Distance to ``region_geojson`` is ≤ ``buffer_km`` km.

    For every passing vessel the function also pre-computes the scoring
    inputs that ``score_candidates`` will consume:

    - ``min_distance_km`` — minimum distance over the release window.
    - ``entry_time`` / ``exit_time`` — first / last qualifying point (ISO-8601 UTC).
    - ``temporal_score`` / ``overlap_minutes`` — from ``_compute_temporal_score``.
    - ``max_gap_minutes`` — from ``_compute_ais_anomaly_score``.
    - ``heading`` / ``angle_diff`` — from ``_compute_trajectory_score``
      (using the default drift bearing of 0° North).
    - ``rw_start`` / ``rw_end`` — stored for re-use in ``score_candidates``.

    When no vessels pass, an empty list is returned without raising.
    Callers should append ``"no_candidates_found_in_region"`` to their
    ``warnings`` list when the result is empty.

    Args:
        ais_result: The result from ``ais_service.query_vessels()``.
        region_geojson: GeoJSON ``dict`` (Polygon or MultiPolygon).
        release_window_start: Start of the probable release window (UTC-aware).
        release_window_end: End of the probable release window (UTC-aware).
        buffer_km: Spatial buffer radius in km (default ``20.0``).

    Returns:
        List of intermediate candidate dicts.  Empty list when no vessel
        qualifies.
    """
    if not ais_result.vessels:
        return []

    try:
        region_geom = shape(region_geojson)
    except Exception as exc:
        logger.error("filter_candidates: failed to parse region_geojson — %s", exc)
        return []

    rw_start = _ensure_utc(release_window_start)
    rw_end = _ensure_utc(release_window_end)
    candidates: list[dict[str, Any]] = []

    for vessel in ais_result.vessels:
        if not vessel.track:
            continue

        qualifying_points: list[tuple[float, float, str]] = []
        min_dist_km: float = float("inf")

        for lon, lat, ts_utc in vessel.track:
            try:
                t = _parse_track_time(ts_utc)
            except Exception:
                continue

            if not (rw_start <= t <= rw_end):
                continue

            pt = Point(lon, lat)
            dist_deg = pt.distance(region_geom)
            dist_km = dist_deg * _KM_PER_DEG

            if dist_km < min_dist_km:
                min_dist_km = dist_km

            if dist_km <= buffer_km:
                qualifying_points.append((lon, lat, ts_utc))

        # Vessel must have at least one point inside the buffer during the window.
        if not qualifying_points:
            continue

        entry_time = qualifying_points[0][2]
        exit_time = qualifying_points[-1][2]

        # Pre-compute scoring inputs.
        temporal_score, overlap_minutes = _compute_temporal_score(
            vessel.track, rw_start, rw_end
        )
        _, max_gap_minutes = _compute_ais_anomaly_score(vessel.track, rw_start, rw_end)
        _, heading, angle_diff = _compute_trajectory_score(
            vessel.track, _DEFAULT_DRIFT_BEARING
        )
        in_buffer_fraction, closest_time, dwell_hours, in_window_count, in_buffer_count = (
            _compute_in_buffer_stats(vessel.track, region_geom, buffer_km, rw_start, rw_end)
        )

        candidates.append(
            {
                "vessel": vessel,
                "min_distance_km": min_dist_km if min_dist_km != float("inf") else 0.0,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "closest_time": closest_time or entry_time,
                "temporal_score": temporal_score,
                "overlap_minutes": overlap_minutes,
                "max_gap_minutes": max_gap_minutes,
                "in_buffer_fraction": in_buffer_fraction,
                "points_near_source": max(in_buffer_count, len(qualifying_points)),
                "dwell_hours": dwell_hours,
                "num_positions": vessel.position_count or in_window_count or len(vessel.track),
                "heading": heading,
                "angle_diff": angle_diff,
                "rw_start": rw_start,
                "rw_end": rw_end,
            }
        )

    logger.debug(
        "filter_candidates: %d / %d vessels passed filters.",
        len(candidates),
        len(ais_result.vessels),
    )
    return candidates


def filter_candidates_nearest(
    ais_result: AISQueryResult,
    region_geojson: dict,
    release_window_start: datetime,
    release_window_end: datetime,
    buffer_km: float = 20.0,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Relaxed fallback: rank the *nearest* vessels when the strict spatial
    + temporal filter yields nothing.

    ``filter_candidates`` drops any vessel without a track point that is both
    inside the release window **and** within ``buffer_km`` of the corridor.
    In sparse offshore areas the live AIS feed often only returns vessels
    further out than that, so the strict filter returns ``[]`` and the analyst
    is left with no suspects at all.

    This function keeps every vessel that has at least one usable track point,
    scores it by minimum distance to the corridor (measured over in-window
    points when available, otherwise over the whole track), and returns the
    ``top_n`` closest as candidate dicts in the same shape ``score_candidates``
    consumes. Callers must disclose the relaxation via a warning.

    Args:
        ais_result: The result from ``ais_service.query_vessels()``.
        region_geojson: GeoJSON ``dict`` (Polygon or MultiPolygon).
        release_window_start: Start of the probable release window (UTC-aware).
        release_window_end: End of the probable release window (UTC-aware).
        top_n: Maximum number of nearest vessels to return (default ``10``).

    Returns:
        List of intermediate candidate dicts, nearest first. Empty only when
        the AIS feed returned no vessels with a usable track.
    """
    if not ais_result.vessels:
        return []

    try:
        region_geom = shape(region_geojson)
    except Exception as exc:
        logger.error("filter_candidates_nearest: failed to parse region_geojson — %s", exc)
        return []

    rw_start = _ensure_utc(release_window_start)
    rw_end = _ensure_utc(release_window_end)
    scored: list[tuple[float, dict[str, Any]]] = []

    for vessel in ais_result.vessels:
        if not vessel.track:
            continue

        in_window: list[tuple[float, float, str]] = []
        min_dist_all = float("inf")
        min_dist_window = float("inf")

        for lon, lat, ts_utc in vessel.track:
            dist_km = Point(lon, lat).distance(region_geom) * _KM_PER_DEG
            if dist_km < min_dist_all:
                min_dist_all = dist_km
            try:
                t = _parse_track_time(ts_utc)
            except Exception:
                continue
            if rw_start <= t <= rw_end:
                in_window.append((lon, lat, ts_utc))
                if dist_km < min_dist_window:
                    min_dist_window = dist_km

        pts = in_window or vessel.track
        min_dist_km = min_dist_window if in_window else min_dist_all
        if min_dist_km == float("inf"):
            continue

        temporal_score, overlap_minutes = _compute_temporal_score(
            vessel.track, rw_start, rw_end
        )
        _, max_gap_minutes = _compute_ais_anomaly_score(vessel.track, rw_start, rw_end)
        _, heading, angle_diff = _compute_trajectory_score(
            vessel.track, _DEFAULT_DRIFT_BEARING
        )
        in_buffer_fraction, closest_time, dwell_hours, in_window_count, in_buffer_count = (
            _compute_in_buffer_stats(vessel.track, region_geom, buffer_km, rw_start, rw_end)
        )

        scored.append(
            (
                min_dist_km,
                {
                    "vessel": vessel,
                    "min_distance_km": min_dist_km,
                    "entry_time": pts[0][2],
                    "exit_time": pts[-1][2],
                    "closest_time": closest_time or pts[0][2],
                    "temporal_score": temporal_score,
                    "overlap_minutes": overlap_minutes,
                    "max_gap_minutes": max_gap_minutes,
                    "in_buffer_fraction": in_buffer_fraction,
                    "points_near_source": in_buffer_count,
                    "dwell_hours": dwell_hours,
                    "num_positions": vessel.position_count or in_window_count or len(vessel.track),
                    "heading": heading,
                    "angle_diff": angle_diff,
                    "rw_start": rw_start,
                    "rw_end": rw_end,
                },
            )
        )

    scored.sort(key=lambda x: x[0])
    logger.info(
        "filter_candidates_nearest: returning %d / %d vessels (nearest %.1f km).",
        min(top_n, len(scored)),
        len(ais_result.vessels),
        scored[0][0] if scored else float("nan"),
    )
    return [cand for _, cand in scored[:top_n]]


def score_candidates(
    candidates: list[dict[str, Any]],
    source_reconstruction_result: dict | None,
    buffer_km: float = 20.0,
    weights: tuple[float, float, float, float] = (0.35, 0.30, 0.25, 0.10),
    *,
    detection_time: datetime | None = None,
    max_distance_km: float = 12.0,
    max_time_diff_hours: float = 96.0,
    min_trajectory_overlap: float = 0.05,
    min_points_near_source: int = 2,
    min_score: float = 12.0,
) -> list[RankedCandidate]:
    """Score, sort, and rank *potential source vessel candidates*.

    Every scored vessel is returned. Each carries a ``qualified`` flag plus
    ``qualification_reasons`` — a vessel "qualifies" only when it clears every
    evidence gate (closest approach, time from detection, trajectory overlap,
    AIS points near the source, minimum score). The API surfaces qualified
    vessels by default and keeps the rest available under "show all".

    Consumes the intermediate dicts from ``filter_candidates`` /
    ``filter_candidates_nearest``, computes four explainable rule-based
    sub-scores — **spatial proximity**, **temporal consistency**,
    **trajectory overlap**, **time spent near region** — combines them with the
    configured weights into a 0–100 ``attribution_score``, assembles a
    ``ScoreBreakdown`` and four evidence strings, and returns a sorted, ranked
    ``list[RankedCandidate]``.

    This is a ranking of consistency with the spill, **not** a determination
    that any vessel caused it.

    Sorting order:
        1. Descending ``attribution_score``.
        2. Ascending ``min_distance_km`` (first tie-breaker).
        3. Ascending ``mmsi`` lexicographic (second tie-breaker).

    Args:
        candidates: Intermediate candidate dicts from the filter functions.
        source_reconstruction_result: Stored source reconstruction JSON; used
            to extract ``drift_bearing`` for the heading-alignment note.
        buffer_km: Spatial buffer radius in km; fed into the spatial sub-score.
        weights: Four-tuple ``(w_spatial, w_temporal, w_trajectory,
            w_time_near_region)`` that must sum to 1.0.
        detection_time: The satellite detection time (UTC). Used to compute
            each candidate's ``time_diff_hours``. Falls back to the release
            window end when ``None``.

    Returns:
        Sorted, ranked ``list[RankedCandidate]``; empty list when input is empty.
    """
    if not candidates:
        return []

    w_spatial, w_temporal, w_trajectory, w_time = weights

    # Extract drift bearing from source reconstruction result if present.
    drift_bearing: float = _DEFAULT_DRIFT_BEARING
    if source_reconstruction_result:
        try:
            drift_bearing = float(
                source_reconstruction_result.get("drift_bearing", _DEFAULT_DRIFT_BEARING)
            )
        except (TypeError, ValueError):
            drift_bearing = _DEFAULT_DRIFT_BEARING

    weight_dict: dict[str, float] = {
        "spatial": w_spatial,
        "temporal": w_temporal,
        "trajectory": w_trajectory,
        "time_near_region": w_time,
    }

    det_ref = _ensure_utc(detection_time) if detection_time is not None else None

    # Collect (sort_key_tuple, RankedCandidate) for sorting before rank assignment.
    scored: list[tuple[float, float, str, RankedCandidate]] = []

    for cand in candidates:
        vessel: VesselRecord = cand["vessel"]
        min_distance_km: float = cand["min_distance_km"]
        entry_time: str | None = cand["entry_time"]
        exit_time: str | None = cand["exit_time"]
        overlap_minutes: float = cand.get("overlap_minutes", 0.0)

        rw_start: datetime = _ensure_utc(cand["rw_start"])
        rw_end: datetime = _ensure_utc(cand["rw_end"])
        window_hours = max((rw_end - rw_start).total_seconds() / 3600.0, 1e-9)

        closest_time: str | None = cand.get("closest_time") or entry_time
        trajectory_overlap = float(cand.get("in_buffer_fraction", 0.0) or 0.0)
        num_positions = int(
            cand.get("num_positions") or vessel.position_count or len(vessel.track) or 0
        )

        # ----------------------------------------------------------------
        # Sub-scores  (all in [0, 1])
        # ----------------------------------------------------------------
        # 1. Spatial proximity — 1.0 at the boundary, 0.0 at buffer_km.
        spatial_score = _compute_spatial_score(min_distance_km, buffer_km)

        # 2. Temporal consistency — share of the lookback window with AIS activity.
        temporal_score, _ = _compute_temporal_score(vessel.track, rw_start, rw_end)

        # 3. Trajectory overlap — fraction of positions inside the search buffer.
        #    Heading alignment with the drift vector is kept for the evidence note.
        _heading_score, heading, angle_diff = _compute_trajectory_score(
            vessel.track, drift_bearing
        )
        trajectory_score = max(0.0, min(1.0, trajectory_overlap))

        # 4. Time spent near region — dwell hours vs the lookback window.
        if vessel.activity_hours is not None:
            time_near_region_hours = float(vessel.activity_hours)
        elif cand.get("dwell_hours") is not None:
            time_near_region_hours = float(cand["dwell_hours"])
        else:
            time_near_region_hours = overlap_minutes / 60.0
        time_near_region_hours = max(0.0, time_near_region_hours)
        time_near_region_score = min(1.0, time_near_region_hours / window_hours)

        # ----------------------------------------------------------------
        # Weighted rule-based attribution score  ∈ [0.0, 100.0]
        # ----------------------------------------------------------------
        raw = (
            w_spatial * spatial_score
            + w_temporal * temporal_score
            + w_trajectory * trajectory_score
            + w_time * time_near_region_score
        )
        attribution_score = round(min(100.0, max(0.0, raw * 100.0)), 1)

        # ----------------------------------------------------------------
        # Time difference from the satellite detection
        # ----------------------------------------------------------------
        ref_time = det_ref or rw_end
        time_diff_hours: float | None = None
        if closest_time:
            try:
                t_closest = _parse_track_time(closest_time)
                time_diff_hours = round(
                    abs((ref_time - t_closest).total_seconds()) / 3600.0, 2
                )
            except Exception:
                time_diff_hours = None

        breakdown = ScoreBreakdown(
            spatial_score=spatial_score,
            temporal_score=temporal_score,
            trajectory_score=trajectory_score,
            time_near_region_score=time_near_region_score,
            weights=weight_dict,
        )

        # ----------------------------------------------------------------
        # Evidence — exactly four strings, one per scoring dimension,
        # each stating the measurement behind the sub-score.
        # ----------------------------------------------------------------
        spatial_evidence = (
            f"Closest approach {min_distance_km:.1f} km from the detected spill / source "
            f"region (search buffer {buffer_km:.0f} km)."
        )
        temporal_evidence = (
            f"Historical AIS activity covered {temporal_score * 100:.0f}% of the "
            f"{window_hours:.0f} h window ending at the satellite detection time."
        )
        traj_note = ""
        if heading is not None and angle_diff is not None:
            traj_note = (
                f" Track heading {heading:.0f}° is within {angle_diff:.0f}° of the "
                f"{drift_bearing:.0f}° drift vector."
            )
        trajectory_evidence = (
            f"{trajectory_overlap * 100:.0f}% of the vessel's {num_positions} AIS "
            f"position(s) fall inside the search buffer.{traj_note}"
        )
        tdiff_txt = (
            f" ({time_diff_hours:.1f} h from detection)"
            if time_diff_hours is not None
            else ""
        )
        time_near_evidence = (
            f"~{time_near_region_hours:.1f} h of activity near the region; closest AIS "
            f"fix at {closest_time or 'unknown'}{tdiff_txt}."
        )
        evidence = [
            spatial_evidence,
            temporal_evidence,
            trajectory_evidence,
            time_near_evidence,
        ]

        track_lonlat: list[tuple[float, float]] = []
        track_points: list[tuple[float, float, str]] = []
        for pt in vessel.track or []:
            try:
                lon_f, lat_f = float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            track_lonlat.append((lon_f, lat_f))
            # Keep the real per-fix timestamp for the ship-route replay slider.
            if len(pt) > 2 and pt[2]:
                track_points.append((lon_f, lat_f, str(pt[2])))

        # ----------------------------------------------------------------
        # Evidence gates — a vessel "qualifies" only when it clears them all.
        # ``qualification_reasons`` lists the concrete measurement behind each
        # gate it passed (empty entries note why it failed).
        # ----------------------------------------------------------------
        points_near_source = int(cand.get("points_near_source", 0) or 0)
        intersects_corridor = min_distance_km <= 0.5
        traj_ok = trajectory_overlap >= min_trajectory_overlap or intersects_corridor

        gate_results = {
            "distance": min_distance_km <= max_distance_km,
            "time": time_diff_hours is not None and time_diff_hours <= max_time_diff_hours,
            "trajectory": traj_ok,
            "points": points_near_source >= min_points_near_source,
            "score": attribution_score >= min_score,
        }
        qualified = all(gate_results.values())

        qualification_reasons: list[str] = []
        if gate_results["distance"]:
            qualification_reasons.append(
                f"Closest approach {min_distance_km:.1f} km "
                f"(within the {max_distance_km:.0f} km gate)"
            )
        if gate_results["points"]:
            qualification_reasons.append(
                f"{points_near_source} AIS fix(es) inside the source corridor buffer"
            )
        if gate_results["trajectory"]:
            if intersects_corridor:
                qualification_reasons.append("Track passes through the source corridor")
            else:
                qualification_reasons.append(
                    f"{trajectory_overlap * 100:.0f}% of the track falls inside the search buffer"
                )
        if gate_results["time"] and time_diff_hours is not None:
            qualification_reasons.append(
                f"Closest AIS fix {time_diff_hours:.0f} h from the satellite detection"
            )
        if gate_results["score"]:
            qualification_reasons.append(f"Composite score {attribution_score:.0f}/100")

        ranked = RankedCandidate(
            rank=1,  # overwritten after sort
            mmsi=vessel.mmsi,
            name=vessel.name,
            imo=vessel.imo,
            vessel_type=vessel.vessel_type,
            flag=vessel.flag,
            attribution_score=attribution_score,
            min_distance_km=min_distance_km,
            entry_time=entry_time,
            exit_time=exit_time,
            closest_timestamp=closest_time,
            time_diff_hours=time_diff_hours,
            num_ais_positions=num_positions,
            trajectory_overlap=round(trajectory_overlap, 4),
            time_near_region_hours=round(time_near_region_hours, 3),
            points_near_source=points_near_source,
            qualified=qualified,
            qualification_reasons=qualification_reasons,
            track=track_lonlat,
            track_points=track_points,
            score_breakdown=breakdown,
            evidence=evidence,
        )

        scored.append((attribution_score, min_distance_km, vessel.mmsi, ranked))

    # Sort: descending score, ascending distance, ascending mmsi.
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    result: list[RankedCandidate] = []
    for rank, (_, _, _, candidate) in enumerate(scored, start=1):
        # RankedCandidate is a Pydantic model — we rebuild with correct rank
        # rather than mutating the frozen-by-default object.
        result.append(candidate.model_copy(update={"rank": rank}))

    return result
