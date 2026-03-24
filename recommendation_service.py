"""
services/recommendation_service.py — Hybrid recommendation engine.
Combines semantic similarity, citation overlap, and shared Neo4j concepts
to produce ranked, explainable paper recommendations.
"""

import logging
from typing import List, Dict, Any, Optional

from app.models.schemas import RecommendedPaper, RecommendationReason
from app.services import embedding_service, graph_service
from app.db.mongodb import get_papers_collection

logger = logging.getLogger(__name__)

# Weight coefficients for the hybrid score
W_SEMANTIC = 0.6
W_CITATION = 0.25
W_CONCEPT = 0.15


async def _get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    col = get_papers_collection()
    return await col.find_one({"paper_id": paper_id})


async def _get_papers_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    col = get_papers_collection()
    cursor = col.find({"paper_id": {"$in": ids}})
    return await cursor.to_list(length=len(ids))


async def get_recommendations(
    paper_id: str,
    top_k: int = 10,
) -> List[RecommendedPaper]:
    """
    Produce hybrid recommendations for a given paper.

    Strategy:
        1. Semantic: FAISS top-K neighbours.
        2. Citation: count papers sharing citation IDs (stored in MongoDB).
        3. Concept: shared Neo4j concept nodes.
    Results are combined via weighted scoring and de-duplicated.
    """
    source = await _get_paper(paper_id)
    if not source:
        logger.warning(f"Recommendation requested for unknown paper: {paper_id}")
        return []

    abstract = source.get("abstract", "")
    source_citations: set = set(source.get("citations", []))

    # ── 1. Semantic neighbours ────────────────────────────────────────────────
    semantic_hits = await embedding_service.semantic_search(abstract, top_k=top_k * 2)
    # Remove the source paper itself
    semantic_hits = [(pid, score) for pid, score in semantic_hits if pid != paper_id]

    semantic_map: Dict[str, float] = {pid: score for pid, score in semantic_hits}

    # ── 2. Gather all candidate papers ───────────────────────────────────────
    candidate_ids = list(semantic_map.keys())
    candidates = await _get_papers_by_ids(candidate_ids)

    # ── 3. Compute per-candidate scores ──────────────────────────────────────
    scored: List[RecommendedPaper] = []

    for candidate in candidates:
        cid = candidate.get("paper_id", "")
        if cid == paper_id:
            continue

        # Semantic similarity
        sem_score = semantic_map.get(cid, 0.0)

        # Citation overlap
        cand_citations: set = set(candidate.get("citations", []))
        shared_citations_count = len(source_citations & cand_citations)
        # Normalise: treat 5+ shared citations as perfect overlap
        citation_score = min(shared_citations_count / 5.0, 1.0)

        # Shared concepts (Neo4j)
        shared_concepts = await graph_service.get_shared_concepts(cid)
        concept_score = min(len(shared_concepts) / 5.0, 1.0)

        # Hybrid weighted score
        hybrid_score = (
            W_SEMANTIC * sem_score
            + W_CITATION * citation_score
            + W_CONCEPT * concept_score
        )

        scored.append(
            RecommendedPaper(
                paper_id=cid,
                title=candidate.get("title", ""),
                abstract=candidate.get("abstract", ""),
                domain=candidate.get("domain", ""),
                year=candidate.get("year"),
                score=round(hybrid_score, 4),
                reason=RecommendationReason(
                    semantic_similarity=round(sem_score, 4),
                    shared_citations=shared_citations_count,
                    shared_concepts=shared_concepts or None,
                ),
            )
        )

    # Sort descending by hybrid score and return top_k
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
