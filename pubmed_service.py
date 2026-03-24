"""
services/pubmed_service.py — Fetches papers from PubMed via NCBI E-utilities.
Two-step process: ESearch (get IDs) → EFetch (get XML records).
"""

import httpx
import xml.etree.ElementTree as ET
from typing import List, Optional
import logging

from app.models.schemas import PaperCreate, PaperSource
from app.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_pubmed_papers(
    query: str,
    max_results: int = 10,
    domain_override: Optional[str] = None,
) -> List[PaperCreate]:
    """
    Fetch papers from PubMed using NCBI E-utilities.

    Steps:
        1. ESearch: get a list of PMIDs matching the query.
        2. EFetch: retrieve full XML records for those PMIDs.
    """
    settings = get_settings()
    base = settings.PUBMED_BASE_URL
    api_key = settings.PUBMED_API_KEY

    common_params = {"api_key": api_key} if api_key else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Step 1: ESearch ───────────────────────────────────────────────────
        search_params = {
            **common_params,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "usehistory": "y",
        }
        search_resp = await client.get(f"{base}/esearch.fcgi", params=search_params)
        search_resp.raise_for_status()

        search_root = ET.fromstring(search_resp.text)
        id_list = [el.text for el in search_root.findall(".//Id") if el.text]

        if not id_list:
            logger.info(f"PubMed: no results for query='{query}'")
            return []

        # ── Step 2: EFetch ────────────────────────────────────────────────────
        fetch_params = {
            **common_params,
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
        }
        fetch_resp = await client.get(f"{base}/efetch.fcgi", params=fetch_params)
        fetch_resp.raise_for_status()

    fetch_root = ET.fromstring(fetch_resp.text)
    articles = fetch_root.findall(".//PubmedArticle")

    papers: List[PaperCreate] = []
    for article in articles:
        paper = _parse_pubmed_article(article, domain_override)
        if paper:
            papers.append(paper)

    logger.info(f"PubMed: fetched {len(papers)} papers for query='{query}'")
    return papers


def _parse_pubmed_article(
    article: ET.Element, domain_override: Optional[str]
) -> Optional[PaperCreate]:
    """Parse a single <PubmedArticle> XML element."""
    try:
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            return None
        pmid = pmid_el.text.strip()

        title_el = article.find(".//ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (p.text or "") for p in abstract_parts if p.text
        ).strip()

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = (author.findtext("LastName") or "").strip()
            fore = (author.findtext("ForeName") or "").strip()
            if last:
                authors.append(f"{fore} {last}".strip())

        # Year
        year_el = (
            article.find(".//PubDate/Year")
            or article.find(".//ArticleDate/Year")
        )
        year = int(year_el.text) if year_el is not None and year_el.text else None

        # MeSH heading as domain
        mesh_el = article.find(".//MeshHeadingList/MeshHeading/DescriptorName")
        domain = domain_override or (
            mesh_el.text if mesh_el is not None else "biomedical"
        )

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        # Citation count is not available via EFetch; default 0
        return PaperCreate(
            paper_id=f"pubmed:{pmid}",
            title=title,
            abstract=abstract,
            authors=authors,
            domain=domain,
            source=PaperSource.PUBMED,
            year=year,
            citation_count=0,
            url=url,
        )
    except Exception as e:
        logger.warning(f"Failed to parse PubMed article: {e}")
        return None
