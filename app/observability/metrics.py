"""
Metrics collector — persists one ExecutionLog per workflow run.

Enhanced with:
  - Per-agent execution trace (model, tokens, latency per step)
  - Progress callback for SSE streaming
  - Provider attribution tracking
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ExecutionLog, ApprovalRecord

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Accumulates token counts, latencies, and scores across agent calls
    in a single workflow run, then flushes to the database at the end.

    Enhanced features:
      - agent_trace: ordered list of per-agent call records
      - progress_callback: optional callable for real-time SSE events
    """

    def __init__(
        self,
        run_id: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.run_id = run_id
        self._log = ExecutionLog(id=str(uuid.uuid4()), run_id=run_id)
        self._token_accumulator: list[dict] = []
        self._latency_accumulator: list[float] = []
        self._start_time = datetime.now(timezone.utc)
        self._agent_trace: list[dict] = []
        self._progress_callback = progress_callback

    # ------------------------------------------------------------------
    # Progress streaming
    # ------------------------------------------------------------------

    def emit_progress(self, event: dict) -> None:
        """Emit a progress event via the callback (for SSE streaming)."""
        if self._progress_callback:
            try:
                self._progress_callback(event)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_input(self, goal: str, persona: str, company: str) -> None:
        self._log.goal    = goal
        self._log.persona = persona
        self._log.company = company

    def record_llm_call(
        self,
        model: str,
        prompt_version: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        agent_name: str = "",
    ) -> None:
        self._token_accumulator.append(
            {"prompt": prompt_tokens, "completion": completion_tokens}
        )
        self._latency_accumulator.append(latency_ms)
        # Track the most recent model and prompt version
        self._log.model_name    = model
        self._log.prompt_version = prompt_version

        # Per-agent trace entry
        if agent_name:
            trace_entry = {
                "agent": agent_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": round(latency_ms, 1),
                "prompt_version": prompt_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._agent_trace.append(trace_entry)

    def record_custom_trace(
        self,
        agent_name: str,
        model: str,
        latency_ms: float,
        details: str = "",
    ) -> None:
        self._latency_accumulator.append(latency_ms)
        self._agent_trace.append({
            "agent": agent_name,
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round(latency_ms, 1),
            "prompt_version": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_output(
        self,
        email_subject: str,
        email_body: str,
        followup_body: str,
        score: float,
        revision_cycles: int,
    ) -> None:
        self._log.email_subject  = email_subject
        self._log.email_body     = email_body
        self._log.followup_body  = followup_body
        self._log.evaluation_score = score
        self._log.revision_cycles  = revision_cycles

    def record_planner(self, planner_output: dict) -> None:
        self._log.planner_output = planner_output

    def record_reviewer(self, reviewer_feedback: dict) -> None:
        self._log.reviewer_feedback = reviewer_feedback

    def record_rag(self, used: bool, chunks: int) -> None:
        self._log.rag_context_used     = used
        self._log.rag_chunks_retrieved = chunks

    def record_status(self, status: str, error: str = "") -> None:
        self._log.status        = status
        self._log.error_message = error

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    async def flush(self, session: AsyncSession) -> ExecutionLog:
        """Aggregate totals and persist to DB."""
        total_prompt      = sum(t["prompt"]     for t in self._token_accumulator)
        total_completion  = sum(t["completion"] for t in self._token_accumulator)

        self._log.prompt_token_count     = total_prompt
        self._log.completion_token_count = total_completion
        self._log.total_token_count      = total_prompt + total_completion
        self._log.latency_ms             = sum(self._latency_accumulator)
        self._log.workflow_duration_ms   = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds() * 1000

        # Store the per-agent trace
        self._log.agent_trace = self._agent_trace

        session.add(self._log)
        await session.commit()
        await session.refresh(self._log)

        logger.info(
            "Metrics flushed | run_id=%s tokens=%d score=%.1f latency=%.0fms agents=%d",
            self.run_id,
            self._log.total_token_count,
            self._log.evaluation_score or 0,
            self._log.workflow_duration_ms,
            len(self._agent_trace),
        )
        return self._log


async def save_approval(
    session: AsyncSession,
    run_id: str,
    decision: str,
    note: str = "",
) -> ApprovalRecord:
    record = ApprovalRecord(
        id=str(uuid.uuid4()),
        run_id=run_id,
        decision=decision,
        reviewer_note=note,
    )
    session.add(record)

    # Also update the execution log status
    from sqlalchemy import select
    result = await session.execute(
        select(ExecutionLog).where(ExecutionLog.run_id == run_id)
    )
    log = result.scalar_one_or_none()
    if log:
        log.status = decision

    await session.commit()
    return record
