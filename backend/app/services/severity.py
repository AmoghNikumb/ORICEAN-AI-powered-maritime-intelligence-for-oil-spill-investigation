"""
Shared severity / confidence bucketing for detections.

Used by both the officer alert pipeline (``alert_service``) and the sanitised
public incident feed (``public_service``) so the two never diverge.

A detection summary is described by three raw numbers pulled from an
``InferenceResult`` / ``ml_result`` dict:

- ``max_confidence``   — highest ``spill_detections[*].max_confidence`` in [0, 1]
- ``positive_tiles``   — ``positive_tile_count``
- ``area_km2``         — summed ``spill_detections[*].area_km2``
"""

from __future__ import annotations

from typing import Any

# Severity thresholds — tuned for the coarse tile classifier. "critical" needs a
# strong, wide detection; "low" is a marginal single-tile hit.
_SEVERITY_RULES = (
    # (label, min_confidence, min_tiles, min_area_km2)
    ("critical", 0.90, 8, 2000.0),
    ("high", 0.75, 4, 200.0),
    ("moderate", 0.60, 2, 0.0),
)


def summarise_detection(ml_result: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce an ``ml_result`` dict to the fields severity bucketing needs.

    Returns ``{"has_detection", "max_confidence", "positive_tiles",
    "area_km2", "detection_count"}``. Safe on ``None`` / malformed input.
    """
    ml_result = ml_result or {}
    detections = ml_result.get("spill_detections") or []
    positive_tiles = int(ml_result.get("positive_tile_count") or 0)

    max_conf = 0.0
    area = 0.0
    for det in detections:
        if not isinstance(det, dict):
            continue
        try:
            max_conf = max(max_conf, float(det.get("max_confidence") or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            area += float(det.get("area_km2") or 0.0)
        except (TypeError, ValueError):
            pass

    return {
        "has_detection": bool(detections) and positive_tiles > 0,
        "max_confidence": round(max_conf, 4),
        "positive_tiles": positive_tiles,
        "area_km2": round(area, 2),
        "detection_count": len(detections),
    }


def classify_severity(max_confidence: float, positive_tiles: int, area_km2: float) -> str:
    """Return ``"critical" | "high" | "moderate" | "low"`` for a detection.

    A bucket matches when the detection meets *any* of that bucket's raw
    thresholds; buckets are checked strongest-first.
    """
    for label, min_conf, min_tiles, min_area in _SEVERITY_RULES:
        if (
            max_confidence >= min_conf
            or positive_tiles >= min_tiles
            or area_km2 >= min_area
        ):
            return label
    return "low"


def confidence_band(max_confidence: float) -> str:
    """Coarse public-facing confidence label (no raw score exposed)."""
    if max_confidence >= 0.85:
        return "high"
    if max_confidence >= 0.6:
        return "moderate"
    return "low"
