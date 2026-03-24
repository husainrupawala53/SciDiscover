"""
db/postgres.py — Async PostgreSQL connection using SQLAlchemy + asyncpg.
Defines the cross_domain_links table and provides session management.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Float, DateTime, func
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── SQLAlchemy Base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Model ─────────────────────────────────────────────────────────────────

from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy import func as sa_func


class CrossDomainLink(Base):
    """Stores detected cross-domain paper similarity links."""
    __tablename__ = "cross_domain_links"

    paper_id_1: str = Column(String, primary_key=True)
    paper_id_2: str = Column(String, primary_key=True)
    similarity_score: float = Column(Float, nullable=False)
    domain_1: str = Column(String, nullable=False)
    domain_2: str = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=sa_func.now())


# ── Engine & Session Factory ──────────────────────────────────────────────────

_engine = None
_session_factory: async_sessionmaker = None


async def connect_postgres() -> None:
    """Create async engine and ensure tables exist."""
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_async_engine(settings.POSTGRES_URL, echo=settings.DEBUG)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    # Auto-create tables for hackathon convenience
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ PostgreSQL connected and tables ensured")


async def close_postgres() -> None:
    """Dispose the engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("PostgreSQL connection closed")


async def get_pg_session() -> AsyncSession:
    """Dependency-injectable async session."""
    async with _session_factory() as session:
        yield session
