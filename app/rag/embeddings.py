"""
Embedding utility — uses Google text-embedding-004 via the google-genai SDK.

768-dim vectors. Falls back to an empty list on failure so RAG never
hard-crashes the workflow.
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
    Generate a 768-dim embedding vector for the given text.
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
