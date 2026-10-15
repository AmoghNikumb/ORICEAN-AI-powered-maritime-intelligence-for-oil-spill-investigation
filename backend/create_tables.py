"""
Run this once to create all tables in the Neon PostgreSQL database.

Usage:
    cd backend
    venv\Scripts\python create_tables.py
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Import after env is loaded
    from app.db.database import engine, Base  # noqa: PLC0415
    # Import models so Base knows about them
    import app.models.investigation  # noqa: F401, PLC0415

    logger.info("Connecting to database…")
    async with engine.begin() as conn:
        logger.info("Creating all tables…")
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Done — all tables created successfully.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
