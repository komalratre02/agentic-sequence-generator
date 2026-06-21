"""
Seed script — ingest a sample company knowledge base into Qdrant.

Run: python -m app.rag.seed_knowledge
"""
import asyncio
import uuid
import logging

from qdrant_client.models import PointStruct

from app.rag.qdrant_client import get_qdrant_client, ensure_collection
from app.rag.embeddings import embed_text
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Sample knowledge base — replace with real data or a CSV/JSON loader
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "source": "acme_profile",
        "text": (
            "Acme Inc is a mid-market SaaS company founded in 2018 specialising in "
            "enterprise workflow automation. They serve 500+ customers across fintech, "
            "healthcare, and logistics verticals. ARR of $22M with 35% YoY growth."
        ),
    },
    {
        "source": "acme_pain_points",
        "text": (
            "Acme's CTO, Marcus Reed, has publicly noted challenges around developer "
            "productivity, CI/CD pipeline reliability, and cloud infrastructure costs "
            "following their AWS migration in Q1 2024. Engineering headcount grew 60% "
            "but velocity didn't scale proportionally."
        ),
    },
    {
        "source": "acme_recent_news",
        "text": (
            "In March 2024, Acme raised a $15M Series B led by Andreessen Horowitz. "
            "The funding is earmarked for product expansion and an enterprise sales push "
            "into the Fortune 500. They also announced a strategic partnership with Salesforce."
        ),
    },
    {
        "source": "acme_tech_stack",
        "text": (
            "Acme's engineering stack is Python/FastAPI on the backend, React on the frontend, "
            "with Kubernetes on AWS EKS. They are heavy users of Datadog for observability "
            "and recently evaluated several AI-assisted code review tools."
        ),
    },
    {
        "source": "generic_cto_pain_points",
        "text": (
            "Common CTO pain points in 2024: managing technical debt while shipping features, "
            "AI/LLM integration strategy, attracting senior engineering talent, "
            "security & compliance for enterprise deals, and justifying infra spend to the board."
        ),
    },
    {
        "source": "generic_saas_industry",
        "text": (
            "B2B SaaS companies in the $10M–$50M ARR range typically face pressure to "
            "reduce CAC, improve NRR above 110%, and demonstrate AI-native capabilities "
            "to stay competitive. Engineering efficiency is a top board-level metric."
        ),
    },
]


async def seed() -> None:
    ok = await ensure_collection()
    if not ok:
        logger.error("Qdrant not available. Cannot seed knowledge base.")
        return

    client = get_qdrant_client()
    points = []

    for doc in KNOWLEDGE_BASE:
        vector = await embed_text(doc["text"])
        if not vector:
            logger.warning("Skipping doc '%s' — embedding failed.", doc["source"])
            continue

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": doc["text"], "source": doc["source"]},
            )
        )

    if points:
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.info("Seeded %d documents into Qdrant collection '%s'.", len(points), settings.qdrant_collection)
    else:
        logger.warning("No documents seeded.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(seed())
