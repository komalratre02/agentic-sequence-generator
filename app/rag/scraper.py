"""
Company Website Scraper — live RAG ingestion for any company URL.

Scrapes a URL with httpx + BeautifulSoup, chunks the text into ~300-word
pieces, embeds each chunk via Gemini, and upserts to Qdrant tagged with
the run_id so retrieval is scoped per-run.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from qdrant_client.models import PointStruct, models

from app.config import get_settings
from app.rag.embeddings import embed_text
from app.rag.qdrant_client import get_qdrant_client, ensure_collection

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_WORD_LIMIT = 300
MAX_PAGES = 5          # Max pages to follow (homepage + internal links)
SCRAPE_TIMEOUT = 15    # seconds per request
MAX_CONTENT_LENGTH = 5_000_000  # 5MB — skip huge pages
USER_AGENT = (
    "Mozilla/5.0 (compatible; AISequenceBot/1.0; "
    "+https://github.com/komalratre02/agentic-sequence-generator)"
)


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate a company URL before scraping.
    Returns (is_valid, cleaned_url_or_error_message).
    """
    url = url.strip()
    if not url:
        return False, "URL is empty."

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Could not parse URL: {url}"

    if not parsed.netloc or "." not in parsed.netloc:
        return False, f"Invalid domain in URL: {url}"

    if parsed.scheme not in ("http", "https"):
        return False, f"URL must use http or https, got: {parsed.scheme}"

    # Block localhost / private IPs (SSRF protection)
    blocked = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
    if parsed.netloc.split(":")[0] in blocked:
        return False, "Cannot scrape localhost or private addresses."

    return True, url

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(html: str) -> str:
    """Extract readable text from HTML, stripping scripts/styles/nav."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_internal_links(html: str, base_url: str) -> list[str]:
    """Return up to MAX_PAGES unique internal links from the page."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only same-domain, http(s), skip anchors/media
        if parsed.netloc != base_domain:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean in seen or clean == base_url:
            continue

        # Skip obvious non-content paths
        skip_patterns = ("/cdn-cgi/", "/wp-json/", "/feed", ".xml", ".pdf",
                         ".png", ".jpg", ".css", ".js", "/tag/", "/category/")
        if any(p in clean.lower() for p in skip_patterns):
            continue

        seen.add(clean)
        links.append(clean)
        if len(links) >= MAX_PAGES - 1:  # -1 because we already have the homepage
            break

    return links


def _chunk_text(text: str, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    """Split text into chunks of approximately max_words words."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if len(chunk.split()) >= 15:  # Skip very small tail chunks
            chunks.append(chunk)
    return chunks


async def _fetch_page(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Fetch a single page, returning HTML or None on failure."""
    try:
        resp = await client.get(
            url,
            follow_redirects=True,
            timeout=SCRAPE_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Scrape HTTP %d for %s", resp.status_code, url)
            return None
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            logger.info("Skipping non-HTML content at %s", url)
            return None
        return resp.text
    except httpx.HTTPError as exc:
        logger.warning("Scrape failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def scrape_and_ingest(
    company_url: str,
    run_id: str,
    *,
    progress_callback=None,
    metrics=None,
) -> int:
    """
    Scrape a company website, chunk content, embed, and upsert into Qdrant.

    Args:
        company_url: The URL to scrape (e.g. "https://stripe.com").
        run_id: Unique run identifier — used to tag Qdrant points.
        progress_callback: Optional callable to emit SSE progress events.

    Returns:
        Number of chunks ingested into Qdrant.
    """
    start_time = time.time()
    # Validate and normalise the URL
    is_valid, result = validate_url(company_url)
    if not is_valid:
        logger.warning("Invalid company URL: %s", result)
        if progress_callback:
            progress_callback({
                "type": "agent_complete",
                "agent": "scraper",
                "label": "Website Scraper",
                "model": "httpx+bs4",
                "chunks": 0,
                "warning": result,
            })
        return 0
    company_url = result

    ok = await ensure_collection()
    if not ok:
        logger.error("Qdrant not available — skipping scrape ingestion.")
        return 0

    qdrant = get_qdrant_client()
    if qdrant is None:
        return 0

    if progress_callback:
        progress_callback({
            "type": "agent_start",
            "agent": "scraper",
            "label": "Website Scraper",
        })

    # ── Fetch pages ──────────────────────────────────────────────────────
    all_text_parts: list[tuple[str, str]] = []  # (source_url, text)

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        # 1. Fetch homepage
        home_html = await _fetch_page(client, company_url)
        if home_html:
            home_text = _clean_text(home_html)
            if home_text:
                all_text_parts.append((company_url, home_text))

            # 2. Follow internal links
            internal_links = _extract_internal_links(home_html, company_url)
            logger.info("Found %d internal links on %s", len(internal_links), company_url)

            # Fetch internal pages concurrently
            tasks = [_fetch_page(client, link) for link in internal_links]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for link, result in zip(internal_links, results):
                if isinstance(result, str) and result:
                    page_text = _clean_text(result)
                    if page_text:
                        all_text_parts.append((link, page_text))

    if not all_text_parts:
        logger.warning("No content scraped from %s", company_url)
        if progress_callback:
            progress_callback({
                "type": "agent_complete",
                "agent": "scraper",
                "label": "Website Scraper",
                "model": "httpx+bs4",
                "chunks": 0,
            })
        return 0

    # ── Chunk all text ───────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for source_url, text in all_text_parts:
        for chunk_text in _chunk_text(text):
            all_chunks.append({"source": source_url, "text": chunk_text})

    logger.info(
        "Scraped %d pages → %d chunks from %s",
        len(all_text_parts), len(all_chunks), company_url,
    )

    # ── Embed & upsert ──────────────────────────────────────────────────
    points: list[PointStruct] = []

    # Embed in parallel for speed
    logger.info("Embedding %d chunks in parallel...", len(all_chunks))
    embed_tasks = [embed_text(chunk["text"]) for chunk in all_chunks]
    vectors = await asyncio.gather(*embed_tasks, return_exceptions=True)

    for chunk, vector in zip(all_chunks, vectors):
        if isinstance(vector, Exception) or not vector:
            logger.warning("Embedding failed for chunk from %s: %s", chunk["source"], vector)
            continue

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "run_id": run_id,
                    "type": "scraped",
                },
            )
        )

    if points:
        await qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        logger.info(
            "Ingested %d chunks into Qdrant for run_id=%s from %s",
            len(points), run_id, company_url,
        )

    if progress_callback:
        progress_callback({
            "type": "agent_complete",
            "agent": "scraper",
            "label": "Website Scraper",
            "model": "httpx+bs4",
            "chunks": len(points),
        })

    if metrics:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_custom_trace(
            agent_name="Scraper",
            model="httpx + Gemini Embed",
            latency_ms=latency_ms,
            details=f"{len(points)} chunks"
        )

    return len(points)
