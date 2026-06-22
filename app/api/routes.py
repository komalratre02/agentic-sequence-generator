"""
FastAPI route definitions for the sequence generator API.

Enhanced with:
  - SSE streaming endpoint for real-time pipeline visualization
  - SmartRouter (multi-provider with circuit breaker)
  - Provider health endpoint for observability
"""
import asyncio
import json
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import get_session, ExecutionLog, ApprovalRecord
from app.observability.metrics import MetricsCollector, save_approval
from app.providers.provider_router import SmartRouter
from app.graph.workflow import run_workflow
from app.rag.scraper import scrape_and_ingest

logger    = logging.getLogger(__name__)
router    = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Singleton router — shared across requests for circuit breaker state
_smart_router: Optional[SmartRouter] = None

def get_router() -> SmartRouter:
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    goal:        str
    persona:     str
    company:     str
    company_url: str = ""


class GenerateResponse(BaseModel):
    run_id:          str
    email_subject:   str
    email_body:      str
    followup_subject: str
    followup_body:   str
    score:           float
    revision_cycles: int
    total_tokens:    int
    latency_ms:      float
    status:          str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/api/generate", response_model=GenerateResponse)
async def generate_sequence(
    req: GenerateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Trigger a full workflow run and return the result."""
    run_id  = str(uuid.uuid4())
    metrics = MetricsCollector(run_id)
    llm     = get_router()

    metrics.record_input(req.goal, req.persona, req.company)

    # Scrape company website if URL provided
    if req.company_url:
        try:
            chunks_ingested = await scrape_and_ingest(req.company_url, run_id, metrics=metrics)
            logger.info("Scraped %d chunks from %s for run %s", chunks_ingested, req.company_url, run_id)
        except Exception as exc:
            logger.warning("Scrape failed for %s: %s — continuing without scraped data.", req.company_url, exc)

    try:
        final_state = await run_workflow(
            goal=req.goal,
            persona=req.persona,
            company=req.company,
            llm=llm,
            metrics=metrics,
            run_id=run_id,
        )
    except ValueError as exc:
        logger.warning("Validation failed: %s", exc)
        metrics.record_status("failed", str(exc))
        await metrics.flush(session)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Workflow failed: %s", exc)
        metrics.record_status("error", str(exc))
        await metrics.flush(session)
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")

    emails = final_state.get("emails", {})
    review = final_state.get("review", {})
    score  = float(review.get("overall_score", 0.0))

    metrics.record_output(
        email_subject=emails.get("email_subject", ""),
        email_body=emails.get("email_body", ""),
        followup_body=emails.get("followup_body", ""),
        score=score,
        revision_cycles=final_state.get("revision_count", 0),
    )
    metrics.record_status("pending")
    log = await metrics.flush(session)

    return GenerateResponse(
        run_id=run_id,
        email_subject=emails.get("email_subject", ""),
        email_body=emails.get("email_body", ""),
        followup_subject=emails.get("followup_subject", ""),
        followup_body=emails.get("followup_body", ""),
        score=score,
        revision_cycles=final_state.get("revision_count", 0),
        total_tokens=log.total_token_count or 0,
        latency_ms=log.workflow_duration_ms or 0,
        status="pending",
    )


# ---------------------------------------------------------------------------
# SSE Streaming endpoint — real-time pipeline progress
# ---------------------------------------------------------------------------

@router.post("/api/generate/stream")
async def generate_stream(
    req: GenerateRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Trigger a workflow run with real-time SSE progress streaming.

    Emits events:
      - agent_start:    when an agent begins execution
      - agent_complete: when an agent finishes (with model, tokens, latency)
      - revision_loop:  when reviewer routes back to writer
      - complete:       final result with score, tokens, run_id
      - error:          if the workflow fails
      - heartbeat:      keepalive during long LLM calls
    """
    run_id = str(uuid.uuid4())
    progress_queue: asyncio.Queue = asyncio.Queue()

    def on_progress(event: dict):
        progress_queue.put_nowait(event)

    metrics = MetricsCollector(run_id, progress_callback=on_progress)
    llm = get_router()
    metrics.record_input(req.goal, req.persona, req.company)

    # ── PARALLEL EXECUTION: Scrape and Plan run concurrently ──
    scrape_task = None
    if req.company_url:
        async def background_scrape():
            try:
                chunks = await scrape_and_ingest(
                    req.company_url, run_id, progress_callback=on_progress, metrics=metrics
                )
                logger.info("Scraped %d chunks from %s for run %s", chunks, req.company_url, run_id)
                return chunks
            except Exception as exc:
                logger.warning("Scrape failed for %s: %s", req.company_url, exc)
                on_progress({
                    "type": "agent_complete",
                    "agent": "scraper",
                    "label": "Website Scraper",
                    "model": "httpx+bs4",
                    "chunks": 0,
                    "warning": f"Scrape error: {exc}",
                })
                return 0
                
        scrape_task = asyncio.create_task(background_scrape())

    async def run_and_save():
        """Execute workflow and persist results."""
        try:
            final_state = await run_workflow(
                goal=req.goal,
                persona=req.persona,
                company=req.company,
                llm=llm,
                metrics=metrics,
                run_id=run_id,
                scrape_task=scrape_task,
            )

            emails = final_state.get("emails", {})
            review = final_state.get("review", {})
            score  = float(review.get("overall_score", 0.0))

            metrics.record_output(
                email_subject=emails.get("email_subject", ""),
                email_body=emails.get("email_body", ""),
                followup_body=emails.get("followup_body", ""),
                score=score,
                revision_cycles=final_state.get("revision_count", 0),
            )
            metrics.record_status("pending")
            log = await metrics.flush(session)

            progress_queue.put_nowait({
                "type": "complete",
                "run_id": run_id,
                "score": score,
                "total_tokens": log.total_token_count or 0,
                "latency_ms": round(log.workflow_duration_ms or 0, 0),
                "revision_cycles": final_state.get("revision_count", 0),
                "provider_health": llm.health_report() if hasattr(llm, "health_report") else [],
            })

        except ValueError as exc:
            metrics.record_status("failed", str(exc))
            await metrics.flush(session)
            progress_queue.put_nowait({
                "type": "error",
                "error_type": "validation",
                "message": str(exc),
            })

        except Exception as exc:
            metrics.record_status("error", str(exc))
            await metrics.flush(session)
            progress_queue.put_nowait({
                "type": "error",
                "error_type": "workflow",
                "message": str(exc),
            })

    async def event_generator():
        """SSE event stream generator."""
        task = asyncio.create_task(run_and_save())

        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=2.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        # Drain remaining events
        while not progress_queue.empty():
            event = progress_queue.get_nowait()
            yield f"data: {json.dumps(event)}\n\n"

        # Ensure task exceptions are raised
        if not task.done():
            task.cancel()
        else:
            exc = task.exception()
            if exc and not any(True for _ in []):  # already handled via queue
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Provider Health endpoint
# ---------------------------------------------------------------------------

@router.get("/api/health/providers")
async def provider_health():
    """Return real-time health metrics for all LLM providers."""
    llm = get_router()
    return {
        "providers": llm.health_report(),
        "active_model": llm.model_name(),
        "active_provider": llm.active_provider_name,
    }


@router.post("/api/approve/{run_id}")
async def approve_sequence(
    run_id: str,
    session: AsyncSession = Depends(get_session),
    note: str = "",
):
    record = await save_approval(session, run_id, "approved", note)
    return {"run_id": run_id, "decision": "approved", "id": record.id}


@router.post("/api/reject/{run_id}")
async def reject_sequence(
    run_id: str,
    session: AsyncSession = Depends(get_session),
    note: str = "",
):
    record = await save_approval(session, run_id, "rejected", note)
    return {"run_id": run_id, "decision": "rejected", "id": record.id}


@router.get("/api/runs", response_model=list[dict])
async def list_runs(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(ExecutionLog).order_by(desc(ExecutionLog.created_at)).limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "run_id":           l.run_id,
            "created_at":       l.created_at.isoformat() if l.created_at else None,
            "goal":             l.goal,
            "persona":          l.persona,
            "company":          l.company,
            "score":            l.evaluation_score,
            "status":           l.status,
            "total_tokens":     l.total_token_count,
            "revision_cycles":  l.revision_cycles,
            "workflow_duration_ms": l.workflow_duration_ms,
        }
        for l in logs
    ]


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(ExecutionLog).where(ExecutionLog.run_id == run_id))
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id":               log.run_id,
        "created_at":           log.created_at.isoformat() if log.created_at else None,
        "goal":                 log.goal,
        "persona":              log.persona,
        "company":              log.company,
        "email_subject":        log.email_subject,
        "email_body":           log.email_body,
        "followup_body":        log.followup_body,
        "evaluation_score":     log.evaluation_score,
        "revision_cycles":      log.revision_cycles,
        "model_name":           log.model_name,
        "prompt_version":       log.prompt_version,
        "total_tokens":         log.total_token_count,
        "estimated_cost_usd":   log.estimated_cost_usd,
        "latency_ms":           log.workflow_duration_ms,
        "rag_context_used":     log.rag_context_used,
        "rag_chunks_retrieved": log.rag_chunks_retrieved,
        "status":               log.status,
        "reviewer_feedback":    log.reviewer_feedback,
        "planner_output":       log.planner_output,
        "agent_trace":          log.agent_trace,
    }


# ---------------------------------------------------------------------------
# HTML / Template routes (Phase 5 Human Approval UI)
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(ExecutionLog).order_by(desc(ExecutionLog.created_at)).limit(10)
    )
    runs = result.scalars().all()

    # Compute stats
    total_result = await session.execute(select(func.count(ExecutionLog.id)))
    total_runs = total_result.scalar() or 0

    approved_result = await session.execute(
        select(func.count(ExecutionLog.id)).where(ExecutionLog.status == "approved")
    )
    approved_count = approved_result.scalar() or 0

    pending_result = await session.execute(
        select(func.count(ExecutionLog.id)).where(ExecutionLog.status == "pending")
    )
    pending_count = pending_result.scalar() or 0

    rejected_result = await session.execute(
        select(func.count(ExecutionLog.id)).where(ExecutionLog.status == "rejected")
    )
    rejected_count = rejected_result.scalar() or 0

    # Provider health
    try:
        llm = get_router()
        provider_health = llm.health_report()
    except Exception:
        provider_health = []

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "runs": runs,
            "total_runs": total_runs,
            "approved_count": approved_count,
            "pending_count": pending_count,
            "rejected_count": rejected_count,
            "provider_health": provider_health,
        },
    )


@router.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse(request=request, name="generate.html", context={"request": request})


@router.get("/review/{run_id}", response_class=HTMLResponse)
async def review_page(
    run_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(ExecutionLog).where(ExecutionLog.run_id == run_id))
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Query the latest approval record for this run
    approval_result = await session.execute(
        select(ApprovalRecord)
        .where(ApprovalRecord.run_id == run_id)
        .order_by(desc(ApprovalRecord.created_at))
        .limit(1)
    )
    approval = approval_result.scalar_one_or_none()

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={"request": request, "log": log, "approval": approval},
    )


@router.post("/review/{run_id}/decision", response_class=RedirectResponse)
async def submit_decision(
    run_id: str,
    decision: str = Form(...),
    note: str = Form(default=""),
    session: AsyncSession = Depends(get_session),
):
    await save_approval(session, run_id, decision, note)
    return RedirectResponse(url=f"/review/{run_id}?decided=1", status_code=303)


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(ExecutionLog).order_by(desc(ExecutionLog.created_at)).limit(50)
    )
    runs = result.scalars().all()
    return templates.TemplateResponse(request=request, name="logs.html", context={"request": request, "runs": runs})
