"""
Writer Agent — generates the outreach email sequence.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Optional

from app.providers.llm_provider import LLMProvider, LLMRequest
from app.observability.prompt_loader import load_prompt, prompt_version_tag
from app.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

PROMPT_NAME = "writer"
PROMPT_VERSION = "v1"


async def run_writer(
    plan: dict[str, Any],
    research: dict[str, Any],
    llm: LLMProvider,
    metrics: MetricsCollector,
    reviewer_feedback: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Run the Writer Agent.

    On first call: generates fresh emails from plan + research.
    On revision calls: incorporates reviewer feedback.
    """
    system_prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    version_tag   = prompt_version_tag(PROMPT_NAME, PROMPT_VERSION)

    plan_text     = json.dumps(plan,     indent=2)
    research_text = json.dumps(research, indent=2)

    user_prompt = (
        f"Campaign Plan:\n{plan_text}\n\n"
        f"Research Brief:\n{research_text}\n\n"
    )

    if reviewer_feedback:
        feedback_text = json.dumps(reviewer_feedback, indent=2)
        user_prompt += (
            f"Previous Review Feedback (fix these issues):\n{feedback_text}\n\n"
            "Rewrite the email sequence addressing ALL weaknesses and improvement suggestions.\n"
        )
    else:
        user_prompt += "Write the email sequence.\n"

    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=2048,
        prompt_version=version_tag,
        response_format={"type": "json_object"},
    )

    logger.info("Writer Agent starting | revision=%s", bool(reviewer_feedback))
    response = await llm.complete(request)

    metrics.record_llm_call(
        model=llm.model_name(),
        prompt_version=version_tag,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        agent_name="writer",
    )

    try:
        emails = json.loads(response.content)
        if isinstance(emails, list):
            logger.warning("Writer returned a list of dicts instead of a single dict. Merging elements...")
            merged = {}
            for item in emails:
                if isinstance(item, dict):
                    merged.update(item)
            emails = merged
        if not isinstance(emails, dict):
            raise ValueError("Parsed JSON is not a dictionary")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Writer returned invalid/uncoercible JSON (%s):\n%s", exc, response.content)
        emails = {
            "email_subject": "Quick question",
            "email_body": response.content[:500],
            "followup_subject": "Following up",
            "followup_body": "Just checking in.",
            "personalization_used": [],
        }

    logger.info("Writer complete | model=%s tokens=%d", llm.model_name(), response.total_tokens)
    return emails
