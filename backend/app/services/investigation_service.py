"""
Service layer for Investigation CRUD and status management.

All database interactions are performed here; the API routes remain thin
wrappers that delegate to these async functions.

Raises
------
ValueError
    Raised (not HTTPException) whenever a requested Investigation is not
    found; the calling route is responsible for converting this to HTTP 404.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.investigation import Investigation, InvestigationStatus
from app.schemas.investigation import (
    InvestigationCreate,
    InvestigationUpdate,
    PaginatedInvestigations,
)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_investigation(
    db: AsyncSession,
    data: InvestigationCreate,
) -> Investigation:
    """
    Persist a new Investigation with ``status=pending``.

    Parameters
    ----------
    db:
        Active async database session.
    data:
        Validated creation payload from the request body.

    Returns
    -------
    Investigation
        The newly created and refreshed ORM instance.
    """
    investigation = Investigation(
        name=data.name,
        aoi_geojson=data.aoi_geojson,
        start_date=data.start_date,
        end_date=data.end_date,
        status=InvestigationStatus.pending,
    )
    db.add(investigation)
    await db.commit()
    await db.refresh(investigation)
    return investigation


# ---------------------------------------------------------------------------
# List (paginated)
# ---------------------------------------------------------------------------


async def list_investigations(
    db: AsyncSession,
    page: int,
    page_size: int,
) -> PaginatedInvestigations:
    """
    Return a paginated list of investigations ordered by ``created_at`` desc.

    Parameters
    ----------
    db:
        Active async database session.
    page:
        1-based page number.
    page_size:
        Number of records per page.

    Returns
    -------
    PaginatedInvestigations
        Pydantic wrapper containing the item list and pagination metadata.
    """
    offset = (page - 1) * page_size

    # Total count query
    count_result = await db.execute(select(func.count()).select_from(Investigation))
    total: int = count_result.scalar_one()

    # Paginated data query
    result = await db.execute(
        select(Investigation)
        .order_by(Investigation.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = list(result.scalars().all())

    return PaginatedInvestigations(
        items=items,  # type: ignore[arg-type]  # Pydantic from_attributes handles ORM → schema
        total=total,
        page=page,
        page_size=page_size,
    )


async def list_investigations_slim(
    db: AsyncSession,
    page: int,
    page_size: int,
) -> PaginatedInvestigations:
    """Like ``list_investigations`` but only pulls small columns from the DB.

    ``ml_result`` is reduced to ``{spill_detections, positive_tile_count}`` and
    ``attribution_result`` is dropped, so this never transfers the base64 tile
    imagery / candidate tracks. Used by dashboard/list views.
    """
    offset = (page - 1) * page_size

    total = (await db.execute(select(func.count()).select_from(Investigation))).scalar_one()

    stmt = (
        select(
            Investigation.id,
            Investigation.name,
            Investigation.aoi_geojson,
            Investigation.start_date,
            Investigation.end_date,
            Investigation.status,
            Investigation.ml_result["spill_detections"],
            Investigation.ml_result["positive_tile_count"],
            Investigation.ml_run_at,
            Investigation.error_message,
            Investigation.source_reconstruction_result,
            Investigation.created_at,
            Investigation.updated_at,
        )
        .order_by(Investigation.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    rows = (await db.execute(stmt)).all()

    items = []
    for r in rows:
        spill_detections, positive_tiles = r[6], r[7]
        ml_result = None
        if spill_detections is not None or positive_tiles is not None:
            ml_result = {
                "spill_detections": spill_detections or [],
                "positive_tile_count": positive_tiles or 0,
            }
        items.append(
            {
                "id": r[0],
                "name": r[1],
                "aoi_geojson": r[2],
                "start_date": r[3],
                "end_date": r[4],
                "status": r[5].value if hasattr(r[5], "value") else r[5],
                "ml_result": ml_result,
                "ml_run_at": r[8],
                "error_message": r[9],
                "source_reconstruction_result": r[10],
                "attribution_result": None,
                "created_at": r[11],
                "updated_at": r[12],
            }
        )

    return PaginatedInvestigations(
        items=items,  # type: ignore[arg-type]
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Get single
# ---------------------------------------------------------------------------


async def get_investigation(
    db: AsyncSession,
    id: str,
) -> Investigation | None:
    """
    Fetch a single Investigation by primary key.

    Parameters
    ----------
    db:
        Active async database session.
    id:
        UUID string of the investigation.

    Returns
    -------
    Investigation | None
        The ORM instance, or ``None`` if not found.
    """
    result = await db.execute(
        select(Investigation).where(Investigation.id == id)
    )
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Update (partial)
# ---------------------------------------------------------------------------


async def update_investigation(
    db: AsyncSession,
    id: str,
    data: InvestigationUpdate,
) -> Investigation:
    """
    Apply a partial update to an existing Investigation.

    Only fields that were explicitly set in the request body (i.e. not
    excluded via ``exclude_unset=True``) will be written to the database.

    Parameters
    ----------
    db:
        Active async database session.
    id:
        UUID string of the investigation to update.
    data:
        Validated partial update payload.

    Returns
    -------
    Investigation
        The updated and refreshed ORM instance.

    Raises
    ------
    ValueError
        If no Investigation with the given ``id`` exists.
    """
    investigation = await get_investigation(db, id)
    if investigation is None:
        raise ValueError(f"Investigation '{id}' not found.")

    updates = data.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(investigation, field, value)

    await db.commit()
    await db.refresh(investigation)
    return investigation


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_investigation(
    db: AsyncSession,
    id: str,
) -> None:
    """
    Delete an Investigation by primary key.

    Parameters
    ----------
    db:
        Active async database session.
    id:
        UUID string of the investigation to delete.

    Raises
    ------
    ValueError
        If no Investigation with the given ``id`` exists.
    """
    investigation = await get_investigation(db, id)
    if investigation is None:
        raise ValueError(f"Investigation '{id}' not found.")

    await db.delete(investigation)
    await db.commit()


# ---------------------------------------------------------------------------
# Set status (used by pipeline stages)
# ---------------------------------------------------------------------------


async def set_status(
    db: AsyncSession,
    id: str,
    status: InvestigationStatus,
    error_message: str | None = None,
) -> Investigation:
    """
    Atomically update the status (and optionally error_message) of an
    Investigation.  Used by the GEE and ML pipeline stages to report
    progress and failure without a full PATCH cycle.

    Parameters
    ----------
    db:
        Active async database session.
    id:
        UUID string of the investigation.
    status:
        The new ``InvestigationStatus`` value.
    error_message:
        Optional human-readable error description (written when
        ``status=failed``).

    Returns
    -------
    Investigation
        The updated and refreshed ORM instance.

    Raises
    ------
    ValueError
        If no Investigation with the given ``id`` exists.
    """
    investigation = await get_investigation(db, id)
    if investigation is None:
        raise ValueError(f"Investigation '{id}' not found.")

    investigation.status = status
    if error_message is not None:
        investigation.error_message = error_message

    await db.commit()
    await db.refresh(investigation)
    return investigation
