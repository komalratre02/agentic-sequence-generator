"""
RAG retrieval — semantic search against the Qdrant knowledge base.
"""
import logging
from dataclasses import dataclass

from app.rag.qdrant_client import get_qdrant_client
from app.rag.embeddings import embed_text
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str


async def retrieve_context(
    query: str,
    top_k: int = 4,
) -> list[RetrievedChunk]:
    """
    Embed the query and retrieve the top-k most relevant chunks from Qdrant.

    Returns an empty list if Qdrant is unavailable or no results match.
    """
    client = get_qdrant_client()
    if client is None:
        logger.info("Qdrant not available — returning empty context.")
        return []

    vector = await embed_text(query)
    if not vector:
        return []

    try:
        results = await client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    score=hit.score,
                    source=payload.get("source", "knowledge_base"),
                )
            )

        logger.info("RAG retrieved %d chunks for query: %s", len(chunks), query[:60])
        return chunks

    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return []


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a readable string for the prompt."""
    if not chunks:
        return "No specific company knowledge available."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Context {i} | relevance={chunk.score:.2f}]\n{chunk.text}")
    return "\n\n".join(parts)
