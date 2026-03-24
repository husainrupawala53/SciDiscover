"""
routers/papers.py — Paper retrieval, summary, connections, and recommendations.
GET  /papers/{id}
GET  /papers/{id}/summary
GET  /papers/{id}/connections
GET  /papers/{id}/recommendations
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.models.schemas import (
    PaperResponse,
    SummaryResponse,
    ConnectionsResponse,
    CrossDomainLink,
    RecommendationResponse,
)
from app.db.mongodb import get_papers_collection
from app.db.postgres import get_pg_session, CrossDomainLink as CDLinkORM
from app.services.summarization_service import get_or_generate_summary
from app.services.recommendation_service import get_recommendations

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["Papers"])


def _doc_to_response(doc: dict) -> PaperResponse:
    """Convert a MongoDB document to a PaperResponse Pydantic model."""
    doc.pop("_id", None)
    return PaperResponse(**doc)


# ── GET /papers/{paper_id} ────────────────────────────────────────────────────

@router.get("/{paper_id:path}", response_model=PaperResponse)
async def get_paper(paper_id: str):
    """Return full metadata for a single paper."""
    collection = get_papers_collection()
    doc = await collection.find_one({"paper_id": paper_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return _doc_to_response(doc)


# ── GET /papers/{paper_id}/summary ───────────────────────────────────────────

@router.get("/{paper_id:path}/summary", response_model=SummaryResponse)
async def get_paper_summary(paper_id: str):
    """
    Return an AI-generated abstractive summary of the paper.
    Results are cached in MongoDB after first generation.
    """
    collection = get_papers_collection()
    doc = await collection.find_one({"paper_id": paper_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

    abstract = doc.get("abstract", "")
    if not abstract:
        raise HTTPException(status_code=422, detail="Paper has no abstract to summarise.")

    summary, cached = await get_or_generate_summary(paper_id, abstract, collection)
    return SummaryResponse(
        paper_id=paper_id,
        title=doc.get("title", ""),
        summary=summary,
        cached=cached,
    )


# ── GET /papers/{paper_id}/connections ───────────────────────────────────────

@router.get("/{paper_id:path}/connections", response_model=ConnectionsResponse)
async def get_paper_connections(
    paper_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_pg_session),
):
    """Return cross-domain papers that share high semantic similarity with this paper."""
    stmt = (
        select(CDLinkORM)
        .where(
            or_(
                CDLinkORM.paper_id_1 == paper_id,
                CDLinkORM.paper_id_2 == paper_id,
            )
        )
        .order_by(CDLinkORM.similarity_score.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    links = [
        CrossDomainLink(
            paper_id_1=row.paper_id_1,
            paper_id_2=row.paper_id_2,
            similarity_score=row.similarity_score,
            domain_1=row.domain_1,
            domain_2=row.domain_2,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return ConnectionsResponse(paper_id=paper_id, connections=links, total=len(links))


# ── GET /papers/{paper_id}/recommendations ────────────────────────────────────

@router.get("/{paper_id:path}/recommendations", response_model=RecommendationResponse)
async def get_paper_recommendations(
    paper_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    """
    Return hybrid recommendations: semantic similarity +
    citation overlap + shared knowledge-graph concepts.
    """
    recommendations = await get_recommendations(paper_id, top_k=top_k)
    return RecommendationResponse(
        source_paper_id=paper_id,
        recommendations=recommendations,
    )
