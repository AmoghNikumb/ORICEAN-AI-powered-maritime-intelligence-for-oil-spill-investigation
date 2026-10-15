"""
Smoke tests for investigation_service.py

Uses an in-memory SQLite database (aiosqlite) so no real PostgreSQL
connection is needed.  The app.core.config module requires several env
vars; we set stubs before importing any app code.

SQLite compatibility note:
  The Investigation model uses server_default=func.now() for created_at,
  which SQLAlchemy translates to the appropriate per-dialect function.
  SQLite renders this as CURRENT_TIMESTAMP, so the column is populated
  correctly in tests.  InvestigationRead.created_at is Optional[datetime]
  for forward compatibility.
"""

import os

# ---------------------------------------------------------------------------
# Provide stub env-vars so config.py / settings load without a real .env
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
os.environ.setdefault("MODEL_PATH", "/dev/null")
os.environ.setdefault("GEOTIFF_EXPORT_DIR", "/tmp")

# ---------------------------------------------------------------------------
# Standard imports (after env stubs are in place)
# ---------------------------------------------------------------------------
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Now it's safe to import app modules
from app.db.database import Base
from app.models.investigation import Investigation, InvestigationStatus  # noqa: F401
from app.schemas.investigation import InvestigationCreate, InvestigationUpdate
from app.services import investigation_service

# ---------------------------------------------------------------------------
# Async test infrastructure
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    """
    Yield a fresh AsyncSession backed by an in-memory SQLite database.
    Creates all tables before each test and drops them after.
    """
    engine = create_async_engine(TEST_DB_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_create(**overrides) -> InvestigationCreate:
    defaults = dict(
        name="Gulf of Oman Test",
        aoi_geojson={"type": "Polygon", "coordinates": [[[55, 22], [56, 22], [56, 23], [55, 23], [55, 22]]]},
        start_date="2024-01-01",
        end_date="2024-01-07",
    )
    defaults.update(overrides)
    return InvestigationCreate(**defaults)


# ---------------------------------------------------------------------------
# Tests: create_investigation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_investigation_status_is_pending(db):
    """create_investigation must set status=pending regardless of input."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    assert inv.status == InvestigationStatus.pending


@pytest.mark.asyncio
async def test_create_investigation_generates_id(db):
    """create_investigation must auto-generate a non-empty string id."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    assert isinstance(inv.id, str)
    assert len(inv.id) > 0


@pytest.mark.asyncio
async def test_create_investigation_persists_fields(db):
    """create_investigation must persist name, aoi_geojson, start/end dates."""
    payload = _sample_create(name="My Spill", start_date="2024-03-01", end_date="2024-03-10")
    inv = await investigation_service.create_investigation(db, payload)
    assert inv.name == "My Spill"
    assert inv.start_date == "2024-03-01"
    assert inv.end_date == "2024-03-10"


@pytest.mark.asyncio
async def test_create_two_investigations_get_different_ids(db):
    """Each investigation must receive a unique id."""
    inv1 = await investigation_service.create_investigation(db, _sample_create(name="A"))
    inv2 = await investigation_service.create_investigation(db, _sample_create(name="B"))
    assert inv1.id != inv2.id


# ---------------------------------------------------------------------------
# Tests: get_investigation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_investigation_returns_none_for_unknown_id(db):
    """get_investigation returns None (not raise) for a non-existent id."""
    result = await investigation_service.get_investigation(db, "non-existent-id")
    assert result is None


@pytest.mark.asyncio
async def test_get_investigation_returns_correct_record(db):
    """get_investigation returns the ORM instance that was just created."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    fetched = await investigation_service.get_investigation(db, inv.id)
    assert fetched is not None
    assert fetched.id == inv.id
    assert fetched.name == inv.name


# ---------------------------------------------------------------------------
# Tests: set_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_status_transitions_to_running_gee(db):
    """set_status must update status on an existing investigation."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    updated = await investigation_service.set_status(db, inv.id, InvestigationStatus.running_gee)
    assert updated.status == InvestigationStatus.running_gee


@pytest.mark.asyncio
async def test_set_status_transitions_to_complete(db):
    """set_status must allow transitioning through the full lifecycle."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    await investigation_service.set_status(db, inv.id, InvestigationStatus.running_gee)
    await investigation_service.set_status(db, inv.id, InvestigationStatus.running_ml)
    updated = await investigation_service.set_status(db, inv.id, InvestigationStatus.complete)
    assert updated.status == InvestigationStatus.complete


@pytest.mark.asyncio
async def test_set_status_records_error_message_on_failed(db):
    """set_status with status=failed and error_message must persist the message."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    updated = await investigation_service.set_status(
        db, inv.id, InvestigationStatus.failed, error_message="GEE quota exceeded"
    )
    assert updated.status == InvestigationStatus.failed
    assert updated.error_message == "GEE quota exceeded"


@pytest.mark.asyncio
async def test_set_status_raises_for_missing_id(db):
    """set_status must raise ValueError when the investigation id does not exist."""
    with pytest.raises(ValueError, match="not found"):
        await investigation_service.set_status(db, "ghost-id", InvestigationStatus.running_gee)


# ---------------------------------------------------------------------------
# Tests: list_investigations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_investigations_empty_db(db):
    """list_investigations returns total=0 and empty items when no records exist."""
    result = await investigation_service.list_investigations(db, page=1, page_size=20)
    assert result.total == 0
    assert result.items == []


@pytest.mark.asyncio
async def test_list_investigations_returns_all_records(db):
    """list_investigations total must equal the number of created investigations."""
    for i in range(3):
        await investigation_service.create_investigation(db, _sample_create(name=f"Inv {i}"))
    result = await investigation_service.list_investigations(db, page=1, page_size=20)
    assert result.total == 3
    assert len(result.items) == 3


@pytest.mark.asyncio
async def test_list_investigations_pagination(db):
    """page_size and page parameters must slice the result set correctly."""
    for i in range(5):
        await investigation_service.create_investigation(db, _sample_create(name=f"Inv {i}"))

    page1 = await investigation_service.list_investigations(db, page=1, page_size=2)
    page2 = await investigation_service.list_investigations(db, page=2, page_size=2)
    page3 = await investigation_service.list_investigations(db, page=3, page_size=2)

    assert len(page1.items) == 2
    assert len(page2.items) == 2
    assert len(page3.items) == 1  # last page — only 1 remaining
    assert page1.total == 5


# ---------------------------------------------------------------------------
# Tests: update_investigation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_investigation_patches_name(db):
    """update_investigation with name only must change only the name."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    updated = await investigation_service.update_investigation(
        db, inv.id, InvestigationUpdate(name="Renamed")
    )
    assert updated.name == "Renamed"
    assert updated.status == InvestigationStatus.pending  # untouched


@pytest.mark.asyncio
async def test_update_investigation_raises_for_missing_id(db):
    """update_investigation must raise ValueError for a non-existent id."""
    with pytest.raises(ValueError, match="not found"):
        await investigation_service.update_investigation(
            db, "missing-id", InvestigationUpdate(name="X")
        )


# ---------------------------------------------------------------------------
# Tests: delete_investigation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_investigation_removes_record(db):
    """delete_investigation must remove the record so get returns None afterwards."""
    inv = await investigation_service.create_investigation(db, _sample_create())
    await investigation_service.delete_investigation(db, inv.id)
    assert await investigation_service.get_investigation(db, inv.id) is None


@pytest.mark.asyncio
async def test_delete_investigation_raises_for_missing_id(db):
    """delete_investigation must raise ValueError for a non-existent id."""
    with pytest.raises(ValueError, match="not found"):
        await investigation_service.delete_investigation(db, "ghost-id")
