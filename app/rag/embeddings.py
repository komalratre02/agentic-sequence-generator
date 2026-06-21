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
    Returns an empty list if embedding fails.
    """
    try:
        client = _get_client()
        # The genai SDK embed is synchronous; run in executor to keep async path clean
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            ),
        )
        # response.embeddings is a list of ContentEmbedding
        embedding = response.embeddings[0].values
        return list(embedding)
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return []
