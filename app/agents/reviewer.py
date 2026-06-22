"""
Reviewer Agent — scores the email sequence and routes for revision or approval.
"""
import json
import logging
from typing import Any

from app.providers.llm_provider import LLMProvider, LLMRequest
from app.observability.prompt_loader import load_prompt, prompt_version_tag
from app.observability.metrics import MetricsCollector
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PROMPT_NAME = "reviewer"
PROMPT_VERSION = "v1"


async def run_reviewer(
    emails: dict[str, Any],
    plan: dict[str, Any],
    llm: LLMProvider,
    metrics: MetricsCollector,
) -> dict[str, Any]:
    """
    Run the Reviewer Agent.

    Returns evaluation dict including overall_score and verdict.
    """
    system_prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    version_tag   = prompt_version_tag(PROMPT_NAME, PROMPT_VERSION)

    emails_text = json.dumps(emails, indent=2)
    plan_text   = json.dumps(plan,   indent=2)

    user_prompt = (
        f"Campaign Plan (context):\n{plan_text}\n\n"
        f"Email Sequence to Evaluate:\n{emails_text}\n\n"
        "Provide your evaluation JSON."
    )

    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=1024,
        prompt_version=version_tag,
        response_format={"type": "json_object"},
    )

    logger.info("Reviewer Agent starting")
    response = await llm.complete(request)

    metrics.record_llm_call(
        model=llm.model_name(),
        prompt_version=version_tag,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        agent_name="reviewer",
    )

    try:
        evaluation = json.loads(response.content)
        if isinstance(evaluation, list):
            logger.warning("Reviewer returned a list instead of a dict. Merging elements...")
            merged = {}
            for item in evaluation:
                if isinstance(item, dict):
                    merged.update(item)
            evaluation = merged
        if not isinstance(evaluation, dict):
            raise ValueError("Parsed JSON is not a dictionary")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Reviewer returned invalid/uncoercible JSON (%s):\n%s", exc, response.content)
        evaluation = {
            "overall_score": 5.0,
            "personalization_score": 5.0,
            "clarity_score": 5.0,
            "relevance_score": 5.0,
            "deliverability_score": 5.0,
            "structure_score": 5.0,
            "strengths": [],
            "weaknesses": ["Could not parse reviewer response"],
            "improvement_suggestions": ["Retry generation"],
            "verdict": "revise",
        }

    score = float(evaluation.get("overall_score", 0))
    threshold = settings.min_score_threshold

    # Enforce verdict based on threshold (don't trust LLM's own verdict blindly)
    evaluation["verdict"] = "pass" if score >= threshold else "revise"

    logger.info(
        "Reviewer complete | score=%.1f verdict=%s model=%s tokens=%d",
        score,
        evaluation["verdict"],
        llm.model_name(),
        response.total_tokens,
    )

    metrics.record_reviewer(evaluation)
    return evaluation
