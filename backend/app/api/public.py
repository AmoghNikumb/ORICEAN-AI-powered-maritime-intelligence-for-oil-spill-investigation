"""
FastAPI router for the sanitised public incident feed.

Routes
------
GET /api/public/incidents        — list public spill incidents
GET /api/public/incidents/{id}   — single public spill incident

No authentication. Only non-sensitive fields are returned (see
``app/services/public_service.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services import public_service

router = APIRouter(prefix="/api/public", tags=["public"])


@router.get("/incidents")
async def list_incidents(db: AsyncSession = Depends(get_db)) -> dict:
    incidents = await public_service.list_public_incidents(db)
    return {"data": incidents}


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    incident = await public_service.get_public_incident(db, incident_id)
    if incident is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Not Found", "detail": "No public incident with that id."},
        )
    return {"data": incident}
