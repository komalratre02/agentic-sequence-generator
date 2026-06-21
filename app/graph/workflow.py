from __future__ import annotations
"""
LangGraph StateGraph Workflow — AI Sequence Generator

Graph topology:
  START → planner → research → writer → reviewer → (conditional)
                                  ↑          |
                                  └──────────┘  (score < threshold)
                                               → END (score >= threshold)

State is typed via WorkflowState TypedDict.

Enhanced features:
  - Progress event emission via MetricsCollector for real-time SSE streaming
  - Each node emits agent_start / agent_complete events
"""
import logging
import uuid
from typing import Any, Literal, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END, START

from app.agents.planner  import run_planner
from app.agents.research import run_research
from app.agents.writer   import run_writer
from app.agents.reviewer import run_reviewer
from app.providers.llm_provider import LLMProvider
from app.observability.metrics  import MetricsCollector
from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Workflow State
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    # Input
    run_id:  str
    goal:    str
    persona: str
    company: str

    # Agent outputs
    plan:     dict[str, Any]
    research: dict[str, Any]
    emails:   dict[str, Any]
    review:   dict[str, Any]

    # Control
    revision_count: int
    max_revisions:  int
    llm:            LLMProvider
    metrics:        MetricsCollector

    # Errors
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Node functions (with progress emission)
# ---------------------------------------------------------------------------

async def planner_node(state: WorkflowState) -> dict[str, Any]:
    state["metrics"].emit_progress({
        "type": "agent_start", "agent": "planner", "label": "Planner Agent",
    })
    plan = await run_planner(
        goal=state["goal"],
        persona=state["persona"],
        company=state["company"],
        llm=state["llm"],
        metrics=state["metrics"],
    )
    state["metrics"].emit_progress({
        "type": "agent_complete", "agent": "planner", "label": "Planner Agent",
        "model": state["llm"].model_name(),
    })
    return {"plan": plan}


async def research_node(state: WorkflowState) -> dict[str, Any]:
    state["metrics"].emit_progress({
        "type": "agent_start", "agent": "research", "label": "Research Agent (RAG)",
    })
    research = await run_research(
        company=state["company"],
        persona=state["persona"],
        llm=state["llm"],
        metrics=state["metrics"],
    )
    state["metrics"].emit_progress({
        "type": "agent_complete", "agent": "research", "label": "Research Agent (RAG)",
        "model": state["llm"].model_name(),
    })
    return {"research": research}


async def writer_node(state: WorkflowState) -> dict[str, Any]:
    revision = state.get("revision_count", 0)
    label = f"Writer Agent (rev {revision})" if revision > 0 else "Writer Agent"
    state["metrics"].emit_progress({
        "type": "agent_start", "agent": "writer", "label": label,
    })
    feedback = state.get("review") if revision > 0 else None
    emails = await run_writer(
        plan=state["plan"],
        research=state["research"],
        llm=state["llm"],
        metrics=state["metrics"],
        reviewer_feedback=feedback,
    )
    state["metrics"].emit_progress({
        "type": "agent_complete", "agent": "writer", "label": label,
        "model": state["llm"].model_name(),
    })
    return {"emails": emails}


async def reviewer_node(state: WorkflowState) -> dict[str, Any]:
    state["metrics"].emit_progress({
        "type": "agent_start", "agent": "reviewer", "label": "Reviewer Agent",
    })
    review = await run_reviewer(
        emails=state["emails"],
        plan=state["plan"],
        llm=state["llm"],
        metrics=state["metrics"],
    )
    new_count = state.get("revision_count", 0) + 1
    score = float(review.get("overall_score", 0))
    verdict = review.get("verdict", "revise")
    state["metrics"].emit_progress({
        "type": "agent_complete", "agent": "reviewer", "label": "Reviewer Agent",
        "model": state["llm"].model_name(),
        "score": score, "verdict": verdict,
    })
    return {"review": review, "revision_count": new_count}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_review(state: WorkflowState) -> Literal["writer", "__end__"]:
    review         = state.get("review", {})
    revision_count = state.get("revision_count", 0)
    max_revisions  = state.get("max_revisions", settings.max_revision_cycles)
    verdict        = review.get("verdict", "revise")
    score          = float(review.get("overall_score", 0))

    if verdict == "pass":
        logger.info("Routing → END | score=%.1f revision_count=%d", score, revision_count)
        return "__end__"

    if revision_count >= max_revisions:
        logger.warning(
            "Max revisions (%d) reached | score=%.1f — ending workflow.", max_revisions, score
        )
        return "__end__"

    logger.info(
        "Routing → writer (revision %d) | score=%.1f verdict=%s",
        revision_count, score, verdict,
    )
    state["metrics"].emit_progress({
        "type": "revision_loop", "revision": revision_count,
        "score": score, "max_revisions": max_revisions,
    })
    return "writer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("planner",  planner_node)
    graph.add_node("research", research_node)
    graph.add_node("writer",   writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START,      "planner")
    graph.add_edge("planner",  "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer",   "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "__end__": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

async def run_workflow(
    goal: str,
    persona: str,
    company: str,
    llm: LLMProvider,
    metrics: MetricsCollector,
    run_id: str | None = None,
) -> WorkflowState:
    """
    Execute the full LangGraph workflow and return the final state.
    """
    run_id = run_id or str(uuid.uuid4())
    workflow = build_workflow()

    initial_state: WorkflowState = {
        "run_id":         run_id,
        "goal":           goal,
        "persona":        persona,
        "company":        company,
        "plan":           {},
        "research":       {},
        "emails":         {},
        "review":         {},
        "revision_count": 0,
        "max_revisions":  settings.max_revision_cycles,
        "llm":            llm,
        "metrics":        metrics,
        "errors":         [],
    }

    logger.info("Workflow START | run_id=%s", run_id)
    final_state = await workflow.ainvoke(initial_state)
    logger.info("Workflow END   | run_id=%s revisions=%d", run_id, final_state.get("revision_count", 0))

    return final_state
