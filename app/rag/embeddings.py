"""
Embedding utility — uses Google Gemini Embedding via the google-genai SDK.

Supports both single-text and batch embedding. Batch mode sends all texts
in a single API call, eliminating rate-limit issues during scraper ingestion.

Falls back to an empty list on failure so RAG never hard-crashes the workflow.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Google's latest production embedding model
EMBEDDING_MODEL = "gemini-embedding-2"
VECTOR_DIM = 3072

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


async def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.
    Includes automatic retries for rate limits (429).
    Returns an empty list if embedding fails.
    """
    client = _get_client()
    loop = asyncio.get_event_loop()

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                ),
            )
            return list(response.embeddings[0].values)

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning("Gemini embedding rate limit hit. Retrying in %.1fs...", sleep_time)
                    await asyncio.sleep(sleep_time)
                    continue

            logger.warning("Embedding failed after %d attempts: %s", attempt + 1, exc)
            return []

    return []


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in a SINGLE API call (batch mode).

    This is the production-grade approach: instead of firing N parallel
    requests (which triggers rate limits), we send one request containing
    all N texts. Google returns all N vectors in a single response.

    Returns a list of vectors in the same order as the input texts.
    Any text that fails to embed will have an empty list [] at its position.
    """
    if not texts:
        return []

    client = _get_client()
    loop = asyncio.get_event_loop()

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                ),
            )
            vectors = [list(emb.values) for emb in response.embeddings]
            logger.info("Batch embedded %d texts in a single API call", len(vectors))
            return vectors

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning("Gemini batch embed rate limit. Retrying in %.1fs...", sleep_time)
                    await asyncio.sleep(sleep_time)
                    continue

            logger.warning("Batch embedding failed after %d attempts: %s", attempt + 1, exc)
            return [[] for _ in texts]

    return [[] for _ in texts]
