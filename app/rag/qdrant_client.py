"""
Qdrant client wrapper — lazy singleton with graceful degradation.

If Qdrant is unavailable, RAG silently falls back to empty context
so the workflow still completes without crashing.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

from app.rag.embeddings import VECTOR_DIM


@lru_cache(maxsize=1)
def get_qdrant_client() -> Optional[AsyncQdrantClient]:
    """Return a cached async Qdrant client, or None if not configured."""
    url = settings.qdrant_url
    if not url:
        logger.warning("Qdrant URL not set — RAG disabled.")
        return None
    return AsyncQdrantClient(
        url=url,
        api_key=settings.qdrant_api_key or None,
        timeout=10,
    )


async def ensure_collection() -> bool:
    """
    Create the collection if it does not exist.
    Returns True on success, False if Qdrant is unreachable.
    """
    client = get_qdrant_client()
    if client is None:
        return False

    collection = settings.qdrant_collection
    try:
        existing = await client.get_collections()
        names = [c.name for c in existing.collections]
        if collection not in names:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", collection, VECTOR_DIM)
        return True
    except Exception as exc:
        logger.warning("Qdrant unavailable: %s — continuing without RAG.", exc)
        return False
