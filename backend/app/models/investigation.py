"""
SQLAlchemy ORM model for the Investigation entity.

Investigations are the top-level unit of work for an oil spill inquiry.
Each investigation tracks an analyst's AOI, date range, pipeline status,
the ML inference results produced by the detection pipeline, the source
reconstruction results from kinematic drift backtracking, and the vessel
attribution results from AIS candidate scoring.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class InvestigationStatus(str, enum.Enum):
    """Lifecycle states for an Investigation record."""

    pending = "pending"
    running_gee = "running_gee"
    running_ml = "running_ml"
    complete = "complete"
    failed = "failed"


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class Investigation(Base):
    """
    Represents a single oil spill investigation.

    Columns
    -------
    id : str (UUID)
        Primary key — generated automatically.
    name : str
        Human-readable investigation name (e.g. "Gulf of Oman, Jan 2024").
    aoi_geojson : dict (JSON/JSONB)
        GeoJSON Polygon representing the analyst's Area of Interest.
    start_date : str
        ISO date string for the start of the imagery search window (YYYY-MM-DD).
    end_date : str
        ISO date string for the end of the imagery search window (YYYY-MM-DD).
    status : InvestigationStatus
        Current pipeline status; starts as ``pending``.
    ml_result : dict | None (JSON/JSONB)
        Serialised ``InferenceResult`` written after ML inference completes.
    ml_run_at : datetime | None
        UTC timestamp of when the ML pipeline last ran.
    error_message : str | None
        Human-readable error description when status is ``failed``.
    source_reconstruction_result : dict | None (JSON/JSONB)
        Serialised ``SourceReconstructionResponse`` written after drift
        backtracking completes.  ``None`` until the source reconstruction
        pipeline has been run for this investigation.
    attribution_result : dict | None (JSON/JSONB)
        Serialised ``AttributionResponse`` written after AIS candidate
        scoring completes.  ``None`` until vessel attribution has been run
        for this investigation.
    created_at : datetime
        UTC timestamp set by the database on INSERT.
    updated_at : datetime | None
        UTC timestamp updated on every Python-side UPDATE via ``onupdate``.
    """

    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    name: Mapped[str] = mapped_column(String, nullable=False)

    aoi_geojson: Mapped[dict] = mapped_column(JSON, nullable=False)

    start_date: Mapped[str] = mapped_column(String, nullable=False)

    end_date: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[InvestigationStatus] = mapped_column(
        Enum(InvestigationStatus, name="investigationstatus"),
        nullable=False,
        default=InvestigationStatus.pending,
    )

    ml_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    ml_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    source_reconstruction_result: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None
    )

    attribution_result: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Investigation id={self.id!r} name={self.name!r} status={self.status}>"
