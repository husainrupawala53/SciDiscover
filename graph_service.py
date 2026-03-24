"""
services/graph_service.py — Neo4j knowledge graph builder and query service.
Creates Paper, Author, Concept, and Domain nodes plus their relationships.
"""

import asyncio
import logging
from typing import List, Dict, Any

from app.db.neo4j_db import run_query
from app.models.schemas import GraphNode, GraphEdge, GraphResponse, PaperCreate

logger = logging.getLogger(__name__)


# ── Node / Relationship Creation ──────────────────────────────────────────────

def _run_in_thread(cypher: str, params: dict = None):
    """Execute a blocking Neo4j query in the default thread-pool."""
    return asyncio.get_event_loop().run_in_executor(
        None, run_query, cypher, params or {}
    )


async def create_paper_graph(paper: PaperCreate) -> None:
    """
    Persist a paper and all related nodes/edges in Neo4j.

    Nodes created:
        - (:Paper)
        - (:Author) for each author
        - (:Domain) for the paper domain

    Relationships:
        - Paper-[:AUTHORED_BY]->Author
        - Paper-[:BELONGS_TO]->Domain
    """
    # Upsert Paper node
    await _run_in_thread(
        """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title,
            p.year  = $year,
            p.url   = $url,
            p.source = $source
        """,
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "url": paper.url or "",
            "source": paper.source,
        },
    )

    # Upsert Domain node and link
    await _run_in_thread(
        """
        MERGE (d:Domain {name: $domain})
        WITH d
        MATCH (p:Paper {paper_id: $paper_id})
        MERGE (p)-[:BELONGS_TO]->(d)
        """,
        {"domain": paper.domain, "paper_id": paper.paper_id},
    )

    # Upsert Author nodes and AUTHORED_BY edges
    for author_name in paper.authors:
        if not author_name.strip():
            continue
        await _run_in_thread(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {paper_id: $paper_id})
            MERGE (p)-[:AUTHORED_BY]->(a)
            """,
            {"name": author_name, "paper_id": paper.paper_id},
        )

    logger.debug(f"Neo4j graph created for paper {paper.paper_id}")


async def add_similarity_relationship(
    paper_id_1: str, paper_id_2: str, score: float
) -> None:
    """Create a SIMILAR_TO edge between two papers."""
    await _run_in_thread(
        """
        MATCH (p1:Paper {paper_id: $pid1})
        MATCH (p2:Paper {paper_id: $pid2})
        MERGE (p1)-[r:SIMILAR_TO]-(p2)
        SET r.score = $score
        """,
        {"pid1": paper_id_1, "pid2": paper_id_2, "score": score},
    )


async def add_citation_relationship(
    citing_id: str, cited_id: str
) -> None:
    """Create a CITES edge between two papers."""
    await _run_in_thread(
        """
        MATCH (p1:Paper {paper_id: $citing})
        MATCH (p2:Paper {paper_id: $cited})
        MERGE (p1)-[:CITES]->(p2)
        """,
        {"citing": citing_id, "cited": cited_id},
    )


# ── Graph Query ───────────────────────────────────────────────────────────────

async def get_paper_subgraph(paper_id: str, depth: int = 2) -> GraphResponse:
    """
    Return the ego-graph of a paper for D3.js visualisation.
    Traverses up to `depth` hops from the given paper node.
    """
    cypher = """
        MATCH path = (p:Paper {paper_id: $paper_id})-[*0..2]-(n)
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS src, endNode(rel) AS tgt, type(rel) AS rel_type,
             properties(rel) AS rel_props,
             nodes(path) AS path_nodes
        UNWIND path_nodes AS node
        RETURN DISTINCT
            id(node)            AS node_id,
            labels(node)[0]     AS node_type,
            properties(node)    AS node_props,
            id(src)             AS src_id,
            id(tgt)             AS tgt_id,
            rel_type,
            rel_props
    """
    raw = await _run_in_thread(cypher, {"paper_id": paper_id})

    nodes_map: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    for row in raw:
        nid = str(row["node_id"])
        if nid not in nodes_map:
            props = row.get("node_props") or {}
            label = props.get("name") or props.get("title") or props.get("paper_id") or nid
            nodes_map[nid] = GraphNode(
                id=nid,
                label=label,
                type=row.get("node_type", "Unknown"),
                properties=props,
            )
        # Edge
        if row.get("src_id") is not None and row.get("tgt_id") is not None:
            edge = GraphEdge(
                source=str(row["src_id"]),
                target=str(row["tgt_id"]),
                relationship=row.get("rel_type", "RELATED"),
                properties=row.get("rel_props") or {},
            )
            edges.append(edge)

    # Deduplicate edges
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e.source, e.target, e.relationship)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    return GraphResponse(nodes=list(nodes_map.values()), edges=unique_edges)


async def get_shared_concepts(paper_id: str) -> List[str]:
    """Return Concept node names connected to this paper (via Authors or Domain)."""
    raw = await _run_in_thread(
        """
        MATCH (p:Paper {paper_id: $paper_id})-[:SHARES_CONCEPT]->(c:Concept)
        RETURN c.name AS concept
        """,
        {"paper_id": paper_id},
    )
    return [r["concept"] for r in raw if r.get("concept")]
