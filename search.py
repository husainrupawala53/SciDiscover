"""
routers/search.py — Keyword and semantic search endpoints.
POST /search/semantic  — FAISS vector similarity search
GET  /search           — MongoDB keyword + filter search (paginated)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pymongo import DESCENDING, ASCENDING

from app.models.schemas import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
    KeywordSearchResponse,
    PaperResponse,
    SortField,
)
from app.services.embedding_service import semantic_search
from app.db.mongodb import get_papers_collection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


# ── POST /search/semantic ─────────────────────────────────────────────────────

@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_vector_search(request: SemanticSearchRequest):
    """
    Embed the query text and retrieve the top-K semantically similar papers
    from the FAISS index. Optionally filter results to a single domain.
    """
    hits = await semantic_search(request.query, top_k=request.top_k * 2)

    if not hits:
        return SemanticSearchResponse(results=[], total=0)

    collection = get_papers_collection()

    # Fetch paper metadata for returned IDs
    paper_ids = [pid for pid, _ in hits]
    score_map = {pid: score for pid, score in hits}

    cursor = collection.find({"paper_id": {"$in": paper_ids}})
    docs = await cursor.to_list(length=len(paper_ids))

    results = []
    for doc in docs:
        pid = doc.get("paper_id", "")
        domain = doc.get("domain", "")

        # Optional domain filter
        if request.domain_filter and domain != request.domain_filter:
            continue

        results.append(
            SemanticSearchResult(
                paper_id=pid,
                title=doc.get("title", ""),
                abstract=doc.get("abstract", ""),
                domain=domain,
                score=round(score_map.get(pid, 0.0), 4),
                year=doc.get("year"),
            )
        )

    # Re-sort by score (order may differ after domain filter)
    results.sort(key=lambda r: r.score, reverse=True)
    results = results[: request.top_k]

    return SemanticSearchResponse(results=results, total=len(results))


# ── GET /search ───────────────────────────────────────────────────────────────

@router.get("", response_model=KeywordSearchResponse)
async def keyword_search(
    q: Optional[str] = Query(default=None, description="Full-text search query"),
    domain: Optional[str] = Query(default=None, description="Filter by domain"),
    year_from: Optional[int] = Query(default=None, ge=1900),
    year_to: Optional[int] = Query(default=None, le=2100),
    author: Optional[str] = Query(default=None),
    sort: SortField = Query(default=SortField.RELEVANCE),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    """
    Keyword + filter search over MongoDB.

    Query Parameters:
        q           — Full-text search (title, abstract, authors)
        domain      — Exact domain match
        year_from   — Lower bound for publication year
        year_to     — Upper bound for publication year
        author      — Case-insensitive substring match on authors list
        sort        — relevance | citation_count | year
        page        — Page number (1-indexed)
        page_size   — Results per page
    """
    collection = get_papers_collection()

    # ── Build MongoDB query ───────────────────────────────────────────────────
    mongo_filter: dict = {}

    if q:
        mongo_filter["$text"] = {"$search": q}

    if domain:
        mongo_filter["domain"] = domain

    if year_from or year_to:
        mongo_filter["year"] = {}
        if year_from:
            mongo_filter["year"]["$gte"] = year_from
        if year_to:
            mongo_filter["year"]["$lte"] = year_to

    if author:
        mongo_filter["authors"] = {"$regex": author, "$options": "i"}

    # ── Sorting ───────────────────────────────────────────────────────────────
    sort_spec = []
    if sort == SortField.RELEVANCE and q:
        sort_spec = [("score", {"$meta": "textScore"})]
    elif sort == SortField.CITATION_COUNT:
        sort_spec = [("citation_count", DESCENDING)]
    elif sort == SortField.YEAR:
        sort_spec = [("year", DESCENDING)]

    # ── Pagination ────────────────────────────────────────────────────────────
    skip = (page - 1) * page_size
    total = await collection.count_documents(mongo_filter)

    projection = None
    if sort == SortField.RELEVANCE and q:
        projection = {"score": {"$meta": "textScore"}}

    cursor = collection.find(mongo_filter, projection)
    if sort_spec:
        cursor = cursor.sort(sort_spec)
    cursor = cursor.skip(skip).limit(page_size)

    docs = await cursor.to_list(length=page_size)

    results = []
    for doc in docs:
        doc.pop("_id", None)
        doc.pop("score", None)          # MongoDB text score — not part of schema
        results.append(PaperResponse(**doc))

    return KeywordSearchResponse(
        results=results,
        total=total,
        page=page,
        page_size=page_size,
    )
