"""
main.py — FastAPI application entry point.

Startup order:
    1. MongoDB  → create text index on papers collection
    2. PostgreSQL → ensure cross_domain_links table exists
    3. Neo4j    → verify connectivity
    4. FAISS    → load or create in-memory index
    5. Mount all routers
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.mongodb import connect_mongodb, close_mongodb, get_papers_collection
from app.db.postgres import connect_postgres, close_postgres
from app.db.neo4j_db import connect_neo4j, close_neo4j
from app.db.vector_db import load_or_create_index
from app.routers import ingest, papers, search, graph, recommendations

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage all database connections and ML model warm-up within the
    FastAPI lifespan context so resources are properly cleaned up.
    """
    logger.info("🚀 SciDiscover API starting up…")

    # ── Connect databases ─────────────────────────────────────────────────────
    await connect_mongodb()

    # Ensure MongoDB full-text index exists on the papers collection
    try:
        col = get_papers_collection()
        await col.create_index(
            [("title", "text"), ("abstract", "text"), ("authors", "text")],
            name="papers_text_index",
            default_language="english",
        )
        logger.info("MongoDB text index ensured on papers collection")
    except Exception as e:
        logger.warning(f"MongoDB index creation skipped (may already exist): {e}")

    await connect_postgres()

    try:
        connect_neo4j()
    except Exception as e:
        logger.warning(f"Neo4j connection failed (graph features disabled): {e}")

    # ── Initialise FAISS ──────────────────────────────────────────────────────
    load_or_create_index(dim=384)   # all-MiniLM-L6-v2 produces 384-dim vectors

    logger.info("✅ All services initialised — API is ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("🛑 SciDiscover API shutting down…")
    await close_mongodb()
    await close_postgres()
    try:
        close_neo4j()
    except Exception:
        pass
    logger.info("Shutdown complete")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SciDiscover API",
    description=(
        "AI-powered cross-domain research paper discovery system.\n\n"
        "Ingest papers from arXiv & PubMed, perform semantic search, "
        "detect cross-domain connections, explore a Neo4j knowledge graph, "
        "and get AI-driven recommendations — all via a single async API."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(papers.router)
app.include_router(search.router)
app.include_router(graph.router)
app.include_router(recommendations.router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Lightweight liveness probe.
    Returns service connectivity status for monitoring tools.
    """
    from app.db.vector_db import get_index_size
    from app.db.mongodb import get_database
    from app.models.schemas import HealthResponse

    services: dict = {}

    # MongoDB
    try:
        await get_database().command("ping")
        services["mongodb"] = "ok"
    except Exception as e:
        services["mongodb"] = f"error: {e}"

    # FAISS
    services["faiss"] = f"ok ({get_index_size()} vectors indexed)"

    # Neo4j (best-effort)
    try:
        from app.db.neo4j_db import get_neo4j_driver
        drv = get_neo4j_driver()
        if drv:
            drv.verify_connectivity()
            services["neo4j"] = "ok"
        else:
            services["neo4j"] = "not connected"
    except Exception as e:
        services["neo4j"] = f"error: {e}"

    # PostgreSQL
    try:
        from app.db.postgres import _engine
        async with _engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as e:
        services["postgres"] = f"error: {e}"

    overall = "healthy" if all("error" not in v for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=services,
    )
