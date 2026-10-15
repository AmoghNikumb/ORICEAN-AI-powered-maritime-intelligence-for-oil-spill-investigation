"""
FastAPI router for the officer case workflow.

Routes
------
GET   /api/officer/cases                     — every detected incident + its case state
GET   /api/officer/cases/{investigation_id}  — one case (created lazily)
PATCH /api/officer/cases/{investigation_id}  — update review / response / report state
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.officer_case import OfficerCaseRead, OfficerCaseUpdate
from app.services import officer_case_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/officer", tags=["officer"])


@router.get("/cases")
async def list_cases(db: AsyncSession = Depends(get_db)) -> dict:
    return {"data": await officer_case_service.list_cases(db)}


@router.get("/cases/{investigation_id}")
async def get_case(investigation_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    summary = await officer_case_service.get_case_summary(db, investigation_id)
    if summary is not None:
        return {"data": summary}
    # Unknown investigation — still hand back a default (unsaved) case shell.
    row = await officer_case_service.get_or_create(db, investigation_id)
    return {"data": OfficerCaseRead.model_validate(row)}


@router.patch("/cases/{investigation_id}")
async def patch_case(
    investigation_id: str,
    body: OfficerCaseUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        row = await officer_case_service.update_case(db, investigation_id, body)
    except Exception as exc:  # pragma: no cover
        logger.exception("officer case update failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Officer Case Error", "detail": str(exc)},
        )
    summary = await officer_case_service.get_case_summary(db, investigation_id)
    return {"data": summary or OfficerCaseRead.model_validate(row)}
