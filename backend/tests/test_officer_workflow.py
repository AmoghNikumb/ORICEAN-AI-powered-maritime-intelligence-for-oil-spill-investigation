"""Unit tests for the officer case workflow + attribution track_points."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
os.environ.setdefault("MODEL_PATH", "/dev/null")
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base
from app.schemas.attribution import VesselRecord
from app.schemas.officer_case import OfficerCaseUpdate
from app.services import officer_case_service
from app.services.attribution_service import score_candidates


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as s:
        yield s
    await engine.dispose()


# ---------------------------------------------------------------------------
# attribution track_points
# ---------------------------------------------------------------------------


def test_candidate_track_points_carry_timestamps():
    rw_start = datetime(2024, 1, 14, tzinfo=timezone.utc)
    rw_end = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    v = VesselRecord(
        mmsi="123456789",
        name="TEST",
        track=[
            (55.5, 22.5, "2024-01-14T06:00:00Z"),
            (55.6, 22.6, "2024-01-14T12:00:00Z"),
            (55.7, 22.7, "2024-01-14T18:00:00Z"),
        ],
    )
    cand = {
        "vessel": v,
        "min_distance_km": 1.0,
        "entry_time": rw_start.isoformat(),
        "exit_time": rw_end.isoformat(),
        "closest_time": rw_start.isoformat(),
        "temporal_score": 0.5,
        "overlap_minutes": 60.0,
        "max_gap_minutes": 5.0,
        "in_buffer_fraction": 0.5,
        "points_near_source": 3,
        "dwell_hours": 2.0,
        "num_positions": 3,
        "heading": 90.0,
        "angle_diff": 5.0,
        "rw_start": rw_start,
        "rw_end": rw_end,
    }
    ranked = score_candidates([cand], source_reconstruction_result=None, detection_time=rw_end)
    assert len(ranked) == 1
    tp = ranked[0].track_points
    assert len(tp) == 3
    assert all(len(p) == 3 and isinstance(p[2], str) for p in tp)
    assert tp[0][2] == "2024-01-14T06:00:00Z"


# ---------------------------------------------------------------------------
# officer_case_service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_seeds_checklist(db):
    row = await officer_case_service.get_or_create(db, "inv-1")
    assert row.review_decision == "pending"
    assert row.response_status == "not_started"
    assert isinstance(row.response_checklist, list) and len(row.response_checklist) == 5
    assert all("label" in it and "done" in it for it in row.response_checklist)
    # idempotent
    again = await officer_case_service.get_or_create(db, "inv-1")
    assert again.investigation_id == "inv-1"


@pytest.mark.asyncio
async def test_update_case_stamps_reviewed_at_on_decision_change(db):
    await officer_case_service.get_or_create(db, "inv-2")
    updated = await officer_case_service.update_case(
        db, "inv-2", OfficerCaseUpdate(review_decision="approved", review_notes="ok")
    )
    assert updated.review_decision == "approved"
    assert updated.reviewed_at is not None
    assert updated.review_notes == "ok"

    # a non-decision update must not move reviewed_at
    stamp = updated.reviewed_at
    again = await officer_case_service.update_case(
        db, "inv-2", OfficerCaseUpdate(response_status="responding")
    )
    assert again.reviewed_at == stamp
    assert again.response_status == "responding"
