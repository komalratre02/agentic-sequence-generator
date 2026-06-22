import asyncio
from app.rag.qdrant_client import get_qdrant_client
from qdrant_client.models import models
from app.config import get_settings

async def main():
    client = get_qdrant_client()
    settings = get_settings()
    try:
        await client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="run_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Payload index created.")
    except Exception as e:
        print(f"Index creation failed or exists: {e}")

asyncio.run(main())
