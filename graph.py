"""
routers/graph.py — Knowledge graph query endpoint for D3.js visualisation.
GET /graph/{paper_id}
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import GraphResponse
from app.services.graph_service import get_paper_subgraph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/{paper_id:path}", response_model=GraphResponse)
async def get_graph(paper_id: str):
    """
    Return the knowledge-graph subgraph (nodes + edges) centred on the
    given paper, up to 2 hops away. Suitable for direct consumption by
    front-end D3.js force-directed visualisations.

    Node types: Paper | Author | Concept | Domain
    Edge types: AUTHORED_BY | BELONGS_TO | CITES | SIMILAR_TO | SHARES_CONCEPT
    """
    try:
        graph = await get_paper_subgraph(paper_id)
    except Exception as e:
        logger.error(f"Graph query failed for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Graph query error: {str(e)}")

    if not graph.nodes:
        raise HTTPException(status_code=404, detail=f"No graph data found for paper '{paper_id}'.")

    return graph
