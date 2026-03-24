"""
services/embedding_service.py — Sentence-Transformer embedding service.
Generates embeddings, manages the FAISS index, and exposes semantic search.
Runs heavy model calls in thread-pool executors to stay async-safe.
"""

import asyncio
import logging
import numpy as np
from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.db import vector_db

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model (heavy — runs once at startup)."""
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def _encode_sync(texts: List[str]) -> np.ndarray:
    """Synchronous embed call (runs in executor)."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


async def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Asynchronously encode a list of texts using sentence-transformers.
    Offloads to a thread-pool so the event loop remains unblocked.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode_sync, texts)


async def index_papers(paper_ids: List[str], abstracts: List[str]) -> None:
    """
    Embed paper abstracts and add to the FAISS index.

    Args:
        paper_ids: Matching paper IDs for each abstract.
        abstracts: Raw abstract texts.
    """
    if not paper_ids:
        return
    vecs = await embed_texts(abstracts)
    vector_db.add_embeddings(paper_ids, vecs)
    logger.info(f"Indexed {len(paper_ids)} papers into FAISS")


async def semantic_search(
    query: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Embed the query and retrieve the top-k similar paper IDs from FAISS.

    Returns:
        List of (paper_id, cosine_score) tuples, sorted descending.
    """
    vecs = await embed_texts([query])
    query_vec = vecs[0]
    results = vector_db.search(query_vec, top_k=top_k)
    return results  # Already sorted by FAISS


async def get_embedding_for_paper(abstract: str) -> np.ndarray:
    """Return the embedding vector for a single paper abstract."""
    vecs = await embed_texts([abstract])
    return vecs[0]
