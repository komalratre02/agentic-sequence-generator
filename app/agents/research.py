"""
Research Agent — retrieves company context from the RAG knowledge base.
"""
import json
import logging
from typing import Any

from app.providers.llm_provider import LLMProvider, LLMRequest
from app.observability.prompt_loader import load_prompt, prompt_version_tag
from app.observability.metrics import MetricsCollector
from app.rag.retrieval import retrieve_context, format_context

logger = logging.getLogger(__name__)

PROMPT_NAME = "research"
PROMPT_VERSION = "v1"


async def run_research(
    company: str,
    persona: str,
    llm: LLMProvider,
    metrics: MetricsCollector,
    run_id: str = "",
) -> dict[str, Any]:
    """
    Run the Research Agent.

    1. Build a search query
    2. Retrieve context from Qdrant (filtered by run_id if scraped data exists)
    3. Synthesise into a research brief via LLM
    """
    system_prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    version_tag   = prompt_version_tag(PROMPT_NAME, PROMPT_VERSION)

    # Step 1: Retrieve RAG context (scoped to this run's scraped data)
    query = f"{company} {persona} pain points challenges industry"
    chunks, debug_reason = await retrieve_context(query, top_k=6, run_id=run_id or None)
    context_text = format_context(chunks)

    if not chunks:
        metrics.record_rag(used=False, chunks=0)
        logger.warning("RAG Diagnostic: %s", debug_reason)
        # We will write this debug reason into the context so the user can see it in the UI
        context_text = f"[DEBUG: RAG Failed - {debug_reason}]\n" + context_text
    else:
        metrics.record_rag(used=True, chunks=len(chunks))

    # Step 2: Synthesise
    user_prompt = (
        f"Target Company: {company}\n"
        f"Target Persona: {persona}\n\n"
        f"Retrieved Knowledge Base Context:\n{context_text}\n\n"
        "Produce the research brief JSON."
    )

    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=1024,
        prompt_version=version_tag,
        response_format={"type": "json_object"},
    )

    logger.info("Research Agent starting | company='%s' persona='%s' chunks=%d", company, persona, len(chunks))
    response = await llm.complete(request)

    metrics.record_llm_call(
        model=llm.model_name(),
        prompt_version=version_tag,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        agent_name="research",
    )

    try:
        brief = json.loads(response.content)
        if isinstance(brief, list):
            logger.warning("Research Agent returned a list instead of a dict. Merging elements...")
            merged = {}
            for item in brief:
                if isinstance(item, dict):
                    merged.update(item)
            brief = merged
        if not isinstance(brief, dict):
            raise ValueError("Parsed JSON is not a dictionary")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Research agent returned invalid/uncoercible JSON (%s):\n%s", exc, response.content)
        brief = {
            "company_summary": f"{company} is a B2B software company.",
            "industry": "SaaS",
            "key_pain_points": ["developer productivity", "scaling engineering"],
            "recent_developments": [],
            "persona_priorities": ["technical excellence", "cost optimisation"],
            "recommended_angle": "engineering efficiency",
            "context_source": "generated",
        }

    brief["_raw_chunks"] = len(chunks)
    logger.info("Research complete | model=%s tokens=%d rag_chunks=%d", llm.model_name(), response.total_tokens, len(chunks))
    return brief
