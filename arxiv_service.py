"""
services/arxiv_service.py — Fetches papers from the arXiv Atom API.
Uses httpx async client; parses the Atom XML feed into normalised PaperCreate objects.
"""

import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional
import logging

from app.models.schemas import PaperCreate, PaperSource
from app.config import get_settings

logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def _parse_entry(entry: ET.Element) -> Optional[PaperCreate]:
    """Convert a single Atom <entry> element to a PaperCreate object."""
    try:
        def tag(name, ns=ATOM_NS):
            return f"{{{ns}}}{name}"

        paper_id_raw = (entry.findtext(tag("id")) or "").strip()
        # arXiv IDs look like: http://arxiv.org/abs/2401.00001v1
        paper_id = paper_id_raw.split("/abs/")[-1].split("v")[0]

        title = (entry.findtext(tag("title")) or "").strip().replace("\n", " ")
        abstract = (entry.findtext(tag("summary")) or "").strip().replace("\n", " ")

        authors = [
            author.findtext(tag("name")) or ""
            for author in entry.findall(tag("author"))
        ]

        # Published date → year
        published = entry.findtext(tag("published")) or ""
        year = int(published[:4]) if len(published) >= 4 else None

        # Category → domain
        primary_cat = entry.find(f"{{{ARXIV_NS}}}primary_category")
        domain = primary_cat.get("term", "unknown") if primary_cat is not None else "unknown"

        url = f"https://arxiv.org/abs/{paper_id}"

        return PaperCreate(
            paper_id=f"arxiv:{paper_id}",
            title=title,
            abstract=abstract,
            authors=authors,
            domain=domain,
            source=PaperSource.ARXIV,
            year=year,
            citation_count=0,
            url=url,
        )
    except Exception as e:
        logger.warning(f"Failed to parse arXiv entry: {e}")
        return None


async def fetch_arxiv_papers(
    query: str,
    max_results: int = 10,
    domain_override: Optional[str] = None,
) -> List[PaperCreate]:
    """
    Query the arXiv API and return a list of PaperCreate objects.

    Args:
        query: Search query string.
        max_results: Maximum number of papers to return.
        domain_override: If set, replaces the arXiv category as domain label.
    """
    settings = get_settings()
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(settings.ARXIV_BASE_URL, params=params)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries = root.findall(f"{{{ATOM_NS}}}entry")

    papers: List[PaperCreate] = []
    for entry in entries:
        paper = _parse_entry(entry)
        if paper:
            if domain_override:
                paper.domain = domain_override
            papers.append(paper)

    logger.info(f"arXiv: fetched {len(papers)} papers for query='{query}'")
    return papers
