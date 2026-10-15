"""
Public incident feed — turns internal Investigation rows into the sanitised
``PublicIncident`` shape.

Only investigations with a real spill detection are exposed, and only the
non-sensitive fields: region label, status, detection date, a centroid, the
merged detection polygon, a severity bucket, a coarse confidence band, and the
area. Everything operational is dropped.

Performance: the query selects only the small ``spill_detections`` JSON element
and ``positive_tile_count`` — never ``tile_results`` (which carries base64 tile
imagery and would make this endpoint transfer tens of MB).
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.investigation import Investigation
from app.schemas.public import PublicIncident
from app.services import severity as severity_mod

logger = logging.getLogger(__name__)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def _polygon_centroid(geojson: dict | None) -> list[float] | None:
    try:
        coords = geojson["coordinates"][0]  # exterior ring of a Polygon
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return [round(sum(xs) / len(xs), 4), round(sum(ys) / len(ys), 4)]
    except Exception:
        return None


def _merged_spill_polygon(detections: list) -> dict | None:
    polys = [d.get("polygon_geojson") for d in detections if isinstance(d, dict) and d.get("polygon_geojson")]
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    rings: list = []
    for p in polys:
        if p.get("type") == "Polygon":
            rings.append(p["coordinates"])
        elif p.get("type") == "MultiPolygon":
            rings.extend(p["coordinates"])
    return {"type": "MultiPolygon", "coordinates": rings} if rings else polys[0]


def _incident_from_parts(
    *, id, name, status, end_date, updated_at, created_at, spill_detections, positive_tiles
) -> PublicIncident | None:
    detections = _as_list(spill_detections)
    positive_tiles = int(positive_tiles or 0)
    if not detections or positive_tiles <= 0:
        return None

    max_conf = 0.0
    area = 0.0
    for d in detections:
        if not isinstance(d, dict):
            continue
        try:
            max_conf = max(max_conf, float(d.get("max_confidence") or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            area += float(d.get("area_km2") or 0.0)
        except (TypeError, ValueError):
            pass

    spill_poly = _merged_spill_polygon(detections)
    return PublicIncident(
        id=id,
        region_label=(name or "Unnamed area").strip(),
        status=str(status.value if hasattr(status, "value") else status),
        detection_date=end_date,
        centroid=_polygon_centroid(spill_poly),
        spill_region_geojson=spill_poly,
        severity=severity_mod.classify_severity(max_conf, positive_tiles, round(area, 2)),
        confidence_band=severity_mod.confidence_band(max_conf),
        area_km2=round(area, 2) or None,
        updated_at=updated_at or created_at,
    )


async def list_public_incidents(db: AsyncSession, limit: int = 200) -> list[PublicIncident]:
    stmt = (
        select(
            Investigation.id,
            Investigation.name,
            Investigation.status,
            Investigation.end_date,
            Investigation.updated_at,
            Investigation.created_at,
            Investigation.ml_result["spill_detections"],
            Investigation.ml_result["positive_tile_count"],
        )
        .order_by(Investigation.created_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    out: list[PublicIncident] = []
    for r in rows:
        inc = _incident_from_parts(
            id=r[0], name=r[1], status=r[2], end_date=r[3],
            updated_at=r[4], created_at=r[5], spill_detections=r[6], positive_tiles=r[7],
        )
        if inc is not None:
            out.append(inc)
    return out


async def get_public_incident(db: AsyncSession, incident_id: str) -> PublicIncident | None:
    stmt = select(
        Investigation.id,
        Investigation.name,
        Investigation.status,
        Investigation.end_date,
        Investigation.updated_at,
        Investigation.created_at,
        Investigation.ml_result["spill_detections"],
        Investigation.ml_result["positive_tile_count"],
    ).where(Investigation.id == incident_id)
    r = (await db.execute(stmt)).first()
    if r is None:
        return None
    return _incident_from_parts(
        id=r[0], name=r[1], status=r[2], end_date=r[3],
        updated_at=r[4], created_at=r[5], spill_detections=r[6], positive_tiles=r[7],
    )
