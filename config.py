"""
config.py — Centralized configuration via environment variables.
Loads from .env using python-dotenv.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "SciDiscover"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── MongoDB ───────────────────────────────────────────────────────────────
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "scidiscover"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/scidiscover"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # ── External APIs ─────────────────────────────────────────────────────────
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PUBMED_API_KEY: str = ""
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    # ── Embedding / ML ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    FAISS_INDEX_PATH: str = "faiss_index.bin"
    FAISS_ID_MAP_PATH: str = "faiss_id_map.json"

    # ── Cross-Domain ──────────────────────────────────────────────────────────
    CROSS_DOMAIN_SIMILARITY_THRESHOLD: float = 0.75

    # ── CORS ──────────────────────────────────────────────────────────────────
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
