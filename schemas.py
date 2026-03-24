"""
models/schemas.py — Pydantic request/response schemas for all API routes.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ── Enumerations ──────────────────────────────────────────────────────────────

class PaperSource(str, Enum):
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class SortField(str, Enum):
    RELEVANCE = "relevance"
    CITATION_COUNT = "citation_count"
    YEAR = "year"


# ── Paper Schemas ─────────────────────────────────────────────────────────────

class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class PaperBase(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    domain: str
    source: PaperSource
    year: Optional[int] = None
    citation_count: int = 0
    url: Optional[str] = None


class PaperCreate(PaperBase):
    paper_id: str


class PaperResponse(PaperBase):
    paper_id: str
    summary: Optional[str] = None
    ingested_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Ingestion Schemas ─────────────────────────────────────────────────────────

class ArxivIngestRequest(BaseModel):
    query: str = Field(..., description="arXiv search query string")
    max_results: int = Field(default=10, ge=1, le=100, description="Max papers to fetch")
    domain: Optional[str] = Field(default=None, description="Override domain label")


class PubMedIngestRequest(BaseModel):
    query: str = Field(..., description="PubMed search query string")
    max_results: int = Field(default=10, ge=1, le=100)
    domain: Optional[str] = None


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    paper_ids: List[str]
    message: str


# ── Search Schemas ────────────────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language or paper-text query")
    top_k: int = Field(default=10, ge=1, le=50)
    domain_filter: Optional[str] = None


class SemanticSearchResult(BaseModel):
    paper_id: str
    title: str
    abstract: str
    domain: str
    score: float
    year: Optional[int] = None


class SemanticSearchResponse(BaseModel):
    results: List[SemanticSearchResult]
    total: int


class KeywordSearchResponse(BaseModel):
    results: List[PaperResponse]
    total: int
    page: int
    page_size: int


# ── Cross-Domain Connection Schemas ──────────────────────────────────────────

class CrossDomainLink(BaseModel):
    paper_id_1: str
    paper_id_2: str
    similarity_score: float
    domain_1: str
    domain_2: str
    created_at: Optional[datetime] = None


class ConnectionsResponse(BaseModel):
    paper_id: str
    connections: List[CrossDomainLink]
    total: int


# ── Graph Schemas ─────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    type: str                        # Paper | Author | Concept | Domain
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str                # AUTHORED_BY | CITES | SIMILAR_TO …
    properties: Dict[str, Any] = {}


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ── Summary Schema ────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    paper_id: str
    title: str
    summary: str
    cached: bool


# ── Recommendation Schemas ────────────────────────────────────────────────────

class RecommendationReason(BaseModel):
    semantic_similarity: Optional[float] = None
    shared_citations: Optional[int] = None
    shared_concepts: Optional[List[str]] = None


class RecommendedPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    domain: str
    year: Optional[int] = None
    score: float
    reason: RecommendationReason


class RecommendationResponse(BaseModel):
    source_paper_id: str
    recommendations: List[RecommendedPaper]


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]
