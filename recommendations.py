"""
routers/recommendations.py — Exposes the cross-domain detection trigger.
POST /recommendations/detect-connections — runs batch cross-domain analysis
"""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.cross_domain_service import detect_cross_domain_connections

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


class DetectResponse(BaseModel):
    message: str
    links_found: int = 0


@router.post("/detect-connections", response_model=DetectResponse)
async def trigger_cross_domain_detection(background_tasks: BackgroundTasks):
    """
    Trigger a full batch cross-domain similarity scan across all indexed papers.
    Runs as a background task so the HTTP response returns immediately.

    After calling this, check /papers/{id}/connections for results.
    """
    try:
        # Run in background so the HTTP request returns instantly
        background_tasks.add_task(_run_detection)
        return DetectResponse(
            message="Cross-domain detection started in background. Check /papers/{id}/connections for results."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_detection():
    """Background coroutine wrapper for cross-domain detection."""
    try:
        count = await detect_cross_domain_connections()
        logger.info(f"Background cross-domain detection complete: {count} links found")
    except Exception as e:
        logger.error(f"Background cross-domain detection failed: {e}")
