import asyncio
from app.rag.scraper import scrape_and_ingest
from app.config import get_settings
async def main():
    print(await scrape_and_ingest("stripe.com", "test_id"))
asyncio.run(main())
