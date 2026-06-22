import asyncio
from app.rag.retrieval import retrieve_context
async def main():
    res = await retrieve_context("stripe", top_k=6)
    print(res)
asyncio.run(main())
