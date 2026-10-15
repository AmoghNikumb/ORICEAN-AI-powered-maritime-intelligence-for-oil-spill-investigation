"""
FastAPI router for the Investigation resource.

Routes are intentionally thin — all business logic lives in
``investigation_service``.  Every successful response is wrapped in
``{"data": ...}`` and every error is wrapped in ``{"error": ..., "detail": ...}``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.investigation import (
    InvestigationCreate,
    InvestigationRead,
    InvestigationUpdate,
    PaginatedInvestigations,
)
from app.services import investigation_service

router = APIRouter(tags=["investigations"])


# ---------------------------------------------------------------------------
# POST /  — create
# ---------------------------------------------------------------------------


@router.post("/", status_code=201)
async def create_investigation(
    body: InvestigationCreate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new Investigation and return it wrapped in ``{"data": ...}``."""
    investigation = await investigation_service.create_investigation(db, body)
    return {"data": InvestigationRead.model_validate(investigation)}


# ---------------------------------------------------------------------------
# GET /  — paginated list
# ---------------------------------------------------------------------------


@router.get("/")
async def list_investigations(
    page: int = 1,
    page_size: int = 20,
    slim: bool = False,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return a paginated list of investigations.

    ``slim=true`` selects only small columns from the DB — ``ml_result`` is
    reduced to ``{spill_detections, positive_tile_count}`` and
    ``attribution_result`` is omitted — so list/dashboard views don't transfer
    the base64 tile imagery. A fraction of the size and far faster.
    """
    if slim:
        paginated = await investigation_service.list_investigations_slim(db, page, page_size)
    else:
        paginated = await investigation_service.list_investigations(db, page, page_size)
    return {"data": paginated}


# ---------------------------------------------------------------------------
# GET /{id}  — single record
# ---------------------------------------------------------------------------


@router.get("/{id}")
async def get_investigation(
    id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Fetch a single Investigation by ID.  Returns 404 if not found."""
    investigation = await investigation_service.get_investigation(db, id)
    if investigation is None:
        return Response(
            content='{"error": "Not Found", "detail": "Investigation not found."}',
            status_code=404,
            media_type="application/json",
        )
    return {"data": InvestigationRead.model_validate(investigation)}


# ---------------------------------------------------------------------------
# PATCH /{id}  — partial update
# ---------------------------------------------------------------------------


@router.patch("/{id}")
async def update_investigation(
    id: str,
    body: InvestigationUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Partially update an Investigation.  Returns 404 if not found."""
    try:
        investigation = await investigation_service.update_investigation(db, id, body)
    except ValueError as exc:
        return Response(
            content=f'{{"error": "Not Found", "detail": "{exc}"}}',
            status_code=404,
            media_type="application/json",
        )
    return {"data": InvestigationRead.model_validate(investigation)}


# ---------------------------------------------------------------------------
# DELETE /{id}  — delete
# ---------------------------------------------------------------------------


@router.delete("/{id}", status_code=204)
async def delete_investigation(
    id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Delete an Investigation.  Returns 204 No Content on success, 404 if not found."""
    try:
        await investigation_service.delete_investigation(db, id)
    except ValueError as exc:
        return Response(
            content=f'{{"error": "Not Found", "detail": "{exc}"}}',
            status_code=404,
            media_type="application/json",
        )
    return Response(status_code=204)
