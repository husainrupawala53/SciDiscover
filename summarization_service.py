"""
services/summarization_service.py — Abstractive summarization using BART.
Runs inference in thread-pool executor to keep the event loop non-blocking.
Caches results back to MongoDB to avoid re-computation.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Optional

from transformers import pipeline, Pipeline

from app.config import get_settings

logger = logging.getLogger(__name__)

# Max tokens the model can handle; we truncate inputs accordingly
MAX_INPUT_TOKENS = 1024
MAX_SUMMARY_TOKENS = 180
MIN_SUMMARY_TOKENS = 56


@lru_cache(maxsize=1)
def _get_pipeline() -> Pipeline:
    """Load and cache the BART summarization pipeline (runs once)."""
    settings = get_settings()
    logger.info(f"Loading summarization model: {settings.SUMMARIZATION_MODEL}")
    return pipeline(
        "summarization",
        model=settings.SUMMARIZATION_MODEL,
        device=-1,          # CPU; set to 0 for GPU
    )


def _summarize_sync(text: str) -> str:
    """Synchronous transformers call (runs in executor)."""
    summarizer = _get_pipeline()
    # Truncate to safe token budget before passing to model
    truncated = text[:3500]
    result = summarizer(
        truncated,
        max_length=MAX_SUMMARY_TOKENS,
        min_length=MIN_SUMMARY_TOKENS,
        do_sample=False,
    )
    return result[0]["summary_text"]


async def summarize_text(text: str) -> str:
    """
    Asynchronously summarize a paper abstract using facebook/bart-large-cnn.

    Offloads the heavy transformer call to a thread-pool.
    """
    if not text or len(text) < 50:
        return text
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, _summarize_sync, text)
    return summary


async def get_or_generate_summary(
    paper_id: str,
    abstract: str,
    papers_collection,          # Motor collection handle
) -> tuple[str, bool]:
    """
    Return (summary_text, was_cached).

    Checks MongoDB first; if no cached summary, generates one and stores it.
    """
    doc = await papers_collection.find_one(
        {"paper_id": paper_id}, {"summary": 1}
    )

    if doc and doc.get("summary"):
        return doc["summary"], True

    summary = await summarize_text(abstract)

    # Persist to MongoDB
    await papers_collection.update_one(
        {"paper_id": paper_id},
        {"$set": {"summary": summary}},
        upsert=True,
    )
    return summary, False
