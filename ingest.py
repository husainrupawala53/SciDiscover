"""
routers/ingest.py — Paper ingestion endpoints.
POST /ingest/arxiv  — Fetch papers from arXiv
POST /ingest/pubmed — Fetch papers from PubMed
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from app.models.schemas import (
    ArxivIngestRequest,
    PubMedIngestRequest,
    IngestResponse,
)
from app.services.arxiv_service import fetch_arxiv_papers
from app.services.pubmed_service import fetch_pubmed_papers
from app.services.embedding_service import index_papers
from app.services.graph_service import create_paper_graph
from app.db.mongodb import get_papers_collection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


async def _store_papers(papers) -> IngestResponse:
    """
    Persist a list of PaperCreate objects to MongoDB, then
    index their abstracts in FAISS and build Neo4j graph nodes.
    """
    collection = get_papers_collection()
    ingested_ids, skipped = [], 0

    for paper in papers:
        doc = paper.model_dump()
        doc["ingested_at"] = datetime.now(timezone.utc)

        result = await collection.update_one(
            {"paper_id": paper.paper_id},
            {"$setOnInsert": doc},
            upsert=True,
        )

        if result.upserted_id:
            ingested_ids.append(paper.paper_id)
        else:
            skipped += 1

    # Batch-embed newly ingested papers
    if ingested_ids:
        new_papers = papers if not skipped else [
            p for p in papers if p.paper_id in set(ingested_ids)
        ]
        await index_papers(
            [p.paper_id for p in new_papers],
            [p.abstract for p in new_papers],
        )
        # Build graph nodes for each new paper
        for p in new_papers:
            try:
                await create_paper_graph(p)
            except Exception as e:
                logger.warning(f"Neo4j graph creation failed for {p.paper_id}: {e}")

    return IngestResponse(
        ingested=len(ingested_ids),
        skipped=skipped,
        paper_ids=ingested_ids,
        message=f"Successfully ingested {len(ingested_ids)} papers, skipped {skipped} duplicates.",
    )


@router.post("/arxiv", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_arxiv(request: ArxivIngestRequest):
    """Fetch papers from arXiv and store them in the database."""
    try:
        papers = await fetch_arxiv_papers(
            query=request.query,
            max_results=request.max_results,
            domain_override=request.domain,
        )
    except Exception as e:
        logger.error(f"arXiv fetch error: {e}")
        raise HTTPException(status_code=502, detail=f"arXiv API error: {str(e)}")

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found for the given query.")

    return await _store_papers(papers)


@router.post("/pubmed", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pubmed(request: PubMedIngestRequest):
    """Fetch papers from PubMed and store them in the database."""
    try:
        papers = await fetch_pubmed_papers(
            query=request.query,
            max_results=request.max_results,
            domain_override=request.domain,
        )
    except Exception as e:
        logger.error(f"PubMed fetch error: {e}")
        raise HTTPException(status_code=502, detail=f"PubMed API error: {str(e)}")

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found for the given query.")

    return await _store_papers(papers)
