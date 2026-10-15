"""
FastAPI router for officer alerting.

Routes
------
GET  /api/alerts            — recent recorded alerts (newest first)
GET  /api/alerts/status     — Twilio readiness (no secrets)
GET  /api/alerts/settings   — current notification settings
PUT  /api/alerts/settings   — update notification settings
POST /api/alerts/test       — send a test SMS to the configured recipients
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.alert import Alert
from app.schemas.alert import (
    AlertRead,
    AlertStatus,
    NotificationSettingsRead,
    NotificationSettingsUpdate,
)
from app.services import alert_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _settings_read(row) -> NotificationSettingsRead:
    return NotificationSettingsRead(
        enabled=bool(row.enabled),
        min_confidence=row.min_confidence,
        min_positive_tiles=row.min_positive_tiles,
        recipients=alert_service.settings_recipients(row),
        updated_at=row.updated_at,
    )


@router.get("")
async def list_alerts(limit: int = 50, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(Alert).order_by(Alert.created_at.desc()).limit(max(1, min(limit, 200)))
    )
    rows = list(result.scalars().all())
    return {"data": [AlertRead.model_validate(r) for r in rows]}


@router.get("/status")
async def alerts_status(db: AsyncSession = Depends(get_db)) -> dict:
    row = await alert_service.get_or_create_settings(db)
    recipients = alert_service.settings_recipients(row)
    status = alert_service.twilio_status(recipients, bool(row.enabled))
    return {"data": AlertStatus(**status)}


@router.get("/settings")
async def get_settings(db: AsyncSession = Depends(get_db)) -> dict:
    row = await alert_service.get_or_create_settings(db)
    return {"data": _settings_read(row)}


@router.put("/settings")
async def update_settings(
    body: NotificationSettingsUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    row = await alert_service.get_or_create_settings(db)
    if body.enabled is not None:
        row.enabled = body.enabled
    if body.min_confidence is not None:
        row.min_confidence = body.min_confidence
    if body.min_positive_tiles is not None:
        row.min_positive_tiles = body.min_positive_tiles
    if body.recipients is not None:
        cleaned = [r.strip() for r in body.recipients if r and r.strip()]
        row.recipients = ",".join(cleaned)
    await db.commit()
    await db.refresh(row)
    return {"data": _settings_read(row)}


@router.post("/test")
async def send_test(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        alert = await alert_service.send_test_alert(db)
    except Exception as exc:  # pragma: no cover
        logger.exception("test alert failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Alert Error", "detail": f"Test alert failed: {exc}"},
        )
    return {"data": AlertRead.model_validate(alert)}
