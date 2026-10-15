"""
Officer case workflow service.

Joins each detected investigation to its ``OfficerCase`` row (review decision,
field-response state, report state) and enriches it with the derived facts the
officer views need — severity, detection date, alert count. Reads project the
small ``ml_result`` sub-keys only; the base64 tile imagery never leaves the DB.
"""

from __future__ import annotations

import datetime as _dt
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert
from app.models.investigation import Investigation
from app.models.officer_case import DEFAULT_CHECKLIST, OfficerCase
from app.schemas.officer_case import OfficerCaseRead, OfficerCaseSummary, OfficerCaseUpdate
from app.services import severity as severity_mod

logger = logging.getLogger(__name__)


def _default_checklist() -> list[dict]:
    return [dict(item) for item in DEFAULT_CHECKLIST]


async def get_or_create(db: AsyncSession, investigation_id: str) -> OfficerCase:
    row = await db.get(OfficerCase, investigation_id)
    if row is None:
        row = OfficerCase(
            investigation_id=investigation_id,
            review_decision="pending",
            response_status="not_started",
            response_checklist=_default_checklist(),
            report_status="draft",
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row


async def update_case(
    db: AsyncSession, investigation_id: str, patch: OfficerCaseUpdate
) -> OfficerCase:
    row = await get_or_create(db, investigation_id)

    data = patch.model_dump(exclude_unset=True)
    if "review_decision" in data and data["review_decision"] != row.review_decision:
        row.reviewed_at = _dt.datetime.utcnow()
    if "response_checklist" in data and data["response_checklist"] is not None:
        data["response_checklist"] = [
            {"label": str(it["label"]), "done": bool(it["done"])}
            for it in data["response_checklist"]
        ]

    for field, value in data.items():
        setattr(row, field, value)

    await db.commit()
    await db.refresh(row)
    return row


async def list_cases(db: AsyncSession) -> list[OfficerCaseSummary]:
    inv_rows = (
        await db.execute(
            select(
                Investigation.id,
                Investigation.name,
                Investigation.status,
                Investigation.end_date,
                Investigation.ml_result["spill_detections"],
                Investigation.ml_result["positive_tile_count"],
                Investigation.source_reconstruction_result.isnot(None),
                Investigation.attribution_result.isnot(None),
                Investigation.created_at,
            ).order_by(Investigation.created_at.desc())
        )
    ).all()

    case_rows = {c.investigation_id: c for c in (await db.execute(select(OfficerCase))).scalars().all()}

    alert_counts = dict(
        (await db.execute(
            select(Alert.investigation_id, func.count())
            .where(Alert.investigation_id.isnot(None))
            .group_by(Alert.investigation_id)
        )).all()
    )

    out: list[OfficerCaseSummary] = []
    for r in inv_rows:
        ml_lean = {"spill_detections": r[4] or [], "positive_tile_count": r[5] or 0}
        s = severity_mod.summarise_detection(ml_lean)
        if not s["has_detection"]:
            continue

        case = case_rows.get(r[0])
        base = (
            OfficerCaseRead.model_validate(case).model_dump()
            if case is not None
            else {
                "investigation_id": r[0],
                "review_decision": "pending",
                "review_notes": None,
                "reviewed_at": None,
                "response_status": "not_started",
                "response_checklist": _default_checklist(),
                "report_status": "draft",
                "created_at": r[8],
                "updated_at": None,
            }
        )
        out.append(
            OfficerCaseSummary(
                **base,
                name=r[1],
                status=str(r[2].value if hasattr(r[2], "value") else r[2]),
                severity=severity_mod.classify_severity(
                    s["max_confidence"], s["positive_tiles"], s["area_km2"]
                ),
                max_confidence=s["max_confidence"],
                positive_tiles=s["positive_tiles"],
                area_km2=s["area_km2"],
                detection_date=r[3],
                has_source_reconstruction=bool(r[6]),
                has_attribution=bool(r[7]),
                alert_count=int(alert_counts.get(r[0], 0)),
            )
        )
    return out


async def get_case_summary(db: AsyncSession, investigation_id: str) -> OfficerCaseSummary | None:
    for c in await list_cases(db):
        if c.investigation_id == investigation_id:
            return c
    # No detection / unknown investigation — still return the raw case if one exists.
    case = await db.get(OfficerCase, investigation_id)
    if case is None:
        return None
    inv = await db.get(Investigation, investigation_id)
    base = OfficerCaseRead.model_validate(case).model_dump()
    return OfficerCaseSummary(
        **base,
        name=(inv.name if inv else investigation_id),
        status=str(inv.status.value if inv and hasattr(inv.status, "value") else (inv.status if inv else "unknown")),
        severity="low",
        max_confidence=0.0,
        positive_tiles=0,
        area_km2=0.0,
        detection_date=(inv.end_date if inv else None),
        has_source_reconstruction=bool(inv and inv.source_reconstruction_result),
        has_attribution=bool(inv and inv.attribution_result),
        alert_count=0,
    )
