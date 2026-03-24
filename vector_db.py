"""
db/vector_db.py — FAISS index management.
Handles index creation, persistence, and similarity search.
"""

import faiss
import numpy as np
import json
import os
import logging
from typing import List, Tuple
from app.config import get_settings

logger = logging.getLogger(__name__)

# In-memory state
_index: faiss.Index = None
_id_map: List[str] = []          # position → paper_id


def _get_paths():
    settings = get_settings()
    return settings.FAISS_INDEX_PATH, settings.FAISS_ID_MAP_PATH


def load_or_create_index(dim: int = 384) -> None:
    """Load existing FAISS index from disk or create a new flat L2 index."""
    global _index, _id_map
    idx_path, map_path = _get_paths()

    if os.path.exists(idx_path) and os.path.exists(map_path):
        _index = faiss.read_index(idx_path)
        with open(map_path, "r") as f:
            _id_map = json.load(f)
        logger.info(f"✅ FAISS index loaded ({_index.ntotal} vectors)")
    else:
        # Inner-product index on L2-normalised vectors = cosine similarity
        _index = faiss.IndexFlatIP(dim)
        _id_map = []
        logger.info("✅ FAISS index created (empty)")


def save_index() -> None:
    """Persist the current FAISS index and id map to disk."""
    idx_path, map_path = _get_paths()
    faiss.write_index(_index, idx_path)
    with open(map_path, "w") as f:
        json.dump(_id_map, f)
    logger.debug("FAISS index saved to disk")


def add_embeddings(paper_ids: List[str], embeddings: np.ndarray) -> None:
    """
    Add new paper embeddings to the FAISS index.
    Skips paper_ids that are already indexed.
    """
    global _id_map
    existing = set(_id_map)
    new_ids, new_vecs = [], []

    for pid, vec in zip(paper_ids, embeddings):
        if pid not in existing:
            new_ids.append(pid)
            new_vecs.append(vec)

    if not new_ids:
        return

    vecs = np.array(new_vecs, dtype=np.float32)
    # L2-normalise so inner-product == cosine similarity
    faiss.normalize_L2(vecs)
    _index.add(vecs)
    _id_map.extend(new_ids)
    save_index()
    logger.info(f"Added {len(new_ids)} vectors to FAISS index")


def search(query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Search for the top-k most similar papers.
    Returns list of (paper_id, score) tuples.
    """
    if _index is None or _index.ntotal == 0:
        return []

    vec = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(vec)
    actual_k = min(top_k, _index.ntotal)
    scores, indices = _index.search(vec, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(_id_map):
            results.append((_id_map[idx], float(score)))
    return results


def get_all_embeddings() -> Tuple[List[str], np.ndarray]:
    """Return all stored embeddings and their paper_ids (for batch similarity)."""
    if _index is None or _index.ntotal == 0:
        return [], np.array([])
    # Reconstruct only works with IndexFlatIP
    vecs = np.zeros((_index.ntotal, _index.d), dtype=np.float32)
    for i in range(_index.ntotal):
        vecs[i] = _index.reconstruct(i)
    return list(_id_map), vecs


def get_index_size() -> int:
    return _index.ntotal if _index else 0
