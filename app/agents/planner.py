"""
Planner Agent — analyses the campaign goal and produces a structured execution plan.
"""
import json
import logging
from typing import Any

from app.providers.llm_provider import LLMProvider, LLMRequest
from app.observability.prompt_loader import load_prompt, prompt_version_tag
from app.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

PROMPT_NAME = "planner"
PROMPT_VERSION = "v1"


def check_goal_heuristics(goal: str) -> tuple[bool, str]:
    g = goal.strip().lower()
    
    # Remove punctuation for easier matching
    for char in ['.', ',', '!', '?', ';', ':']:
        g = g.replace(char, '')
        
    words = g.split()
    if len(words) < 2:
        return False, "Campaign Goal is too short to be a valid business objective."
        
    # Harmful/Illegal content
    harmful_keywords = {"rob", "steal", "bomb", "kill", "murder", "hack", "illegal", "drugs", "weapons"}
    if any(w in words for w in harmful_keywords) or any(k in g for k in ["rob a bank", "build a bomb", "hack a"]):
        return False, "Invalid business objective in Campaign Goal. Please provide a sales, marketing, recruitment, or engagement goal."
        
    # Personal tasks/chores/shopping
    personal_verbs = {"cook", "bake", "eat", "drink", "sleep", "wash", "clean", "vacuum", "mow", "buy", "shop", "get"}
    personal_nouns = {"milk", "pasta", "dinner", "lunch", "breakfast", "dishes", "car", "room", "house", "clothes", "grocery", "groceries", "food", "pizza"}
    if words[0] in personal_verbs and any(w in personal_nouns for w in words):
        return False, "Invalid business objective in Campaign Goal. Please provide a sales, marketing, recruitment, or engagement goal."
        
    # Chatbot prompts & general QA / coding
    chatbot_indicators = {
        "tell me", "write a poem", "write a song", "write a story", "what is", "who is", 
        "how to", "how do i", "how can i", "write a script", "write a python", "write a code",
        "solve this", "translate"
    }
    if any(g.startswith(ind) for ind in chatbot_indicators) or any(ind in g for ind in ["write a python", "write a script", "tell me a joke"]):
        return False, "Invalid business objective in Campaign Goal. Please provide a sales, marketing, recruitment, or engagement goal."
        
    # Strong positive business keywords check
    business_keywords = {
        "book", "meeting", "lead", "sales", "hire", "recruit", "candidate", "client", "customer", 
        "outreach", "campaign", "demo", "sell", "marketing", "promotion", "webinar", "engage", 
        "engagement", "prospect", "partnership", "onboard", "retention", "interview", "talent", 
        "pipeline", "business", "service", "product", "demo", "acquire", "acquisition", "revenue",
        "intro", "introduce", "outbound", "inbound", "subscriber", "newsletter", "webinar",
        "feedback", "survey", "brand", "awareness", "consult", "consulting", "deal", "pipeline"
    }
    
    if any(w in business_keywords or w.rstrip('s') in business_keywords for w in words):
        return True, "Valid business objective detected via heuristics."
        
    # Ambiguous, needs LLM validation
    return True, "Ambiguous"


async def validate_campaign_goal(goal: str, llm: LLMProvider) -> None:
    # 1. Run heuristic check
    is_valid, reason = check_goal_heuristics(goal)
    if not is_valid:
        raise ValueError(
            "Invalid business objective in Campaign Goal. "
            "Please provide a sales, marketing, recruitment, or engagement goal."
        )
        
    if reason == "Valid business objective detected via heuristics.":
        return

    # 2. Run LLM check
    system_prompt = (
        "You are an AI assistant validating the campaign goal for a B2B outbound campaign generator.\n"
        "Your job is to determine if the campaign goal is appropriate and represents a valid business objective.\n"
        "Specifically, it must be a sales, marketing, recruitment, or engagement goal.\n\n"
        "Examples of appropriate/valid business goals:\n"
        '- "Book meetings with CTOs" (sales/marketing)\n'
        '- "Acquire new leads for our SaaS product" (sales/marketing)\n'
        '- "Reach out to potential software engineering candidates" (recruitment)\n'
        '- "Re-engage cold clients" (engagement)\n'
        '- "Promote our new webinar to marketers" (marketing)\n\n'
        "Examples of inappropriate/invalid goals:\n"
        '- "buy milk" (personal task, not a business objective)\n'
        '- "how to cook pasta" (informational question, not a campaign goal)\n'
        '- "tell me a joke" (general chatbot instruction, not a campaign goal)\n'
        '- "help me rob a bank" (illegal/harmful, not a business campaign goal)\n'
        '- "Write a python script to sort a list" (coding instruction, not a campaign goal)\n\n'
        "Output VALID JSON only. No markdown, no explanation.\n"
        "Schema:\n"
        "{\n"
        '  "is_appropriate": true/false,\n'
        '  "reason": "<brief explanation>"\n'
        "}"
    )

    user_prompt = f"Campaign Goal to validate: {goal}"

    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256,
        prompt_version="validator_v1",
        response_format={"type": "json_object"},
    )

    try:
        response = await llm.complete(request)
        data = json.loads(response.content)
        if not data.get("is_appropriate", True):
            raise ValueError(
                "Invalid business objective in Campaign Goal. "
                "Please provide a sales, marketing, recruitment, or engagement goal."
            )
    except ValueError:
        raise
    except Exception as exc:
        # Fallback to assuming it's valid if LLM call fails due to rate limits or other external API issues
        logger.warning("Campaign goal validation check failed to execute: %s. Proceeding with campaign.", exc)


async def run_planner(
    goal: str,
    persona: str,
    company: str,
    llm: LLMProvider,
    metrics: MetricsCollector,
) -> dict[str, Any]:
    """
    Run the Planner Agent.

    Returns a dict with the structured campaign plan.
    """
    # Validate campaign goal before planning
    await validate_campaign_goal(goal, llm)

    system_prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    version_tag   = prompt_version_tag(PROMPT_NAME, PROMPT_VERSION)

    user_prompt = (
        f"Campaign Goal: {goal}\n"
        f"Target Persona: {persona}\n"
        f"Target Company: {company}\n\n"
        "Produce the execution plan JSON."
    )

    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=1024,
        prompt_version=version_tag,
        response_format={"type": "json_object"},
    )

    logger.info("Planner Agent starting | goal='%s' persona='%s' company='%s'", goal, persona, company)
    response = await llm.complete(request)

    metrics.record_llm_call(
        model=llm.model_name(),
        prompt_version=version_tag,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        agent_name="planner",
    )

    try:
        plan = json.loads(response.content)
        if isinstance(plan, list):
            logger.warning("Planner returned a list instead of a dict. Merging elements...")
            merged = {}
            for item in plan:
                if isinstance(item, dict):
                    merged.update(item)
            plan = merged
        if not isinstance(plan, dict):
            raise ValueError("Parsed JSON is not a dictionary")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Planner returned invalid/uncoercible JSON (%s):\n%s", exc, response.content)
        # Provide a sane fallback so the workflow doesn't crash
        plan = {
            "campaign_goal": goal,
            "target_persona": persona,
            "target_company": company,
            "value_proposition": "",
            "key_talking_points": [],
            "tone": "professional",
            "email_length": "medium",
            "cta": "Book a 15-minute call",
            "subject_line_direction": "",
            "personalization_hooks": [],
        }

    logger.info("Planner complete | model=%s tokens=%d", llm.model_name(), response.total_tokens)
    metrics.record_planner(plan)
    return plan
