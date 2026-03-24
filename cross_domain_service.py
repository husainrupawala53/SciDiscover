"""
services/cross_domain_service.py — Batch cross-domain connection detection.

After papers are embedded, this service:
  1. Fetches all embeddings from FAISS.
  2. Batches cosine similarity comparisons ONLY across different domains.
  3. Flags pairs above the configured threshold.
  4. Persists them to the PostgreSQL cross_domain_links table.
  5. Optionally adds SIMILAR_TO edges to Neo4j.

Designed to be called as a background task after a large ingestion run.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Dict

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.vector_db import get_all_embeddings
from app.db.mongodb import get_papers_collection
from app.db.postgres import CrossDomainLink, _session_factory
from app.services.graph_service import add_similarity_relationship

logger = logging.getLogger(__name__)


async def _fetch_domain_map() -> Dict[str, str]:
    """Return a dict of {paper_id: domain} from MongoDB."""
    col = get_papers_collection()
    cursor = col.find({}, {"paper_id": 1, "domain": 1, "_id": 0})
    docs = await cursor.to_list(length=10_000)
    return {d["paper_id"]: d.get("domain", "unknown") for d in docs if "paper_id" in d}


def _cosine_similarity_batch(vecs: np.ndarray) -> np.ndarray:
    """
    Compute the full NxN cosine similarity matrix for a set of
    already L2-normalised vectors.

    Since FAISS IndexFlatIP stores L2-normalised vectors, we can simply
    compute the dot-product matrix.
    """
    # vecs is (N, D), already normalised
    return vecs @ vecs.T


async def detect_cross_domain_connections(
    batch_size: int = 500,
) -> int:
    """
    Scan all indexed paper embeddings, detect cross-domain pairs above
    the similarity threshold, and persist them to PostgreSQL.

    Returns:
        Total number of new cross-domain links stored.
    """
    settings = get_settings()
    threshold = settings.CROSS_DOMAIN_SIMILARITY_THRESHOLD

    paper_ids, vecs = get_all_embeddings()
    if len(paper_ids) < 2:
        logger.info("Not enough papers in FAISS index to compute cross-domain links.")
        return 0

    domain_map = await _fetch_domain_map()

    n = len(paper_ids)
    logger.info(f"Computing cross-domain similarities for {n} papers (threshold={threshold})")

    # Compute full similarity matrix (NxN)
    # For large datasets, consider chunked computation to avoid OOM
    sim_matrix = _cosine_similarity_batch(vecs)

    new_links: List[dict] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score < threshold:
                continue

            pid1 = paper_ids[i]
            pid2 = paper_ids[j]
            d1 = domain_map.get(pid1, "unknown")
            d2 = domain_map.get(pid2, "unknown")

            # Only cross-domain pairs
            if d1 == d2:
                continue

            new_links.append(
                {
                    "paper_id_1": pid1,
                    "paper_id_2": pid2,
                    "similarity_score": round(score, 6),
                    "domain_1": d1,
                    "domain_2": d2,
                    "created_at": datetime.now(timezone.utc),
                }
            )

    if not new_links:
        logger.info("No new cross-domain connections found above threshold.")
        return 0

    # ── Persist to PostgreSQL in batches ──────────────────────────────────────
    total_inserted = 0
    async with _session_factory() as session:
        for i in range(0, len(new_links), batch_size):
            chunk = new_links[i : i + batch_size]
            stmt = (
                pg_insert(CrossDomainLink)
                .values(chunk)
                .on_conflict_do_nothing(index_elements=["paper_id_1", "paper_id_2"])
            )
            result = await session.execute(stmt)
            await session.commit()
            total_inserted += result.rowcount

    logger.info(f"Persisted {total_inserted} cross-domain links to PostgreSQL")

    # ── Add SIMILAR_TO edges in Neo4j (best-effort) ───────────────────────────
    for link in new_links[:200]:        # cap Neo4j writes for hackathon demo
        try:
            await add_similarity_relationship(
                link["paper_id_1"], link["paper_id_2"], link["similarity_score"]
            )
        except Exception as e:
            logger.debug(f"Neo4j SIMILAR_TO edge skipped: {e}")

    return total_inserted
