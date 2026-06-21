"""
Database models + async engine setup.

Tables:
- execution_logs   → one row per workflow run
- approval_records → human approve/reject decisions
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, Boolean, JSON
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id                   = Column(String, primary_key=True)
    run_id               = Column(String, index=True, nullable=False)
    created_at           = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Input
    goal                 = Column(Text)
    persona              = Column(Text)
    company              = Column(Text)

    # Output
    email_subject        = Column(Text)
    email_body           = Column(Text)
    followup_body        = Column(Text)
    evaluation_score     = Column(Float)
    revision_cycles      = Column(Integer, default=0)

    # Observability
    model_name           = Column(String)
    prompt_version       = Column(String)       # e.g. "writer_v1"
    prompt_token_count   = Column(Integer)
    completion_token_count = Column(Integer)
    total_token_count    = Column(Integer)
    estimated_cost_usd   = Column(Float)
    latency_ms           = Column(Float)
    workflow_duration_ms = Column(Float)

    # Research / RAG
    rag_context_used     = Column(Boolean, default=False)
    rag_chunks_retrieved = Column(Integer, default=0)

    # Status
    status               = Column(String, default="pending")  # pending | approved | rejected
    error_message        = Column(Text)

    # Raw JSON snapshots
    planner_output       = Column(JSON)
    reviewer_feedback    = Column(JSON)

    # Per-agent execution trace (new — shows model/tokens/latency per agent step)
    agent_trace          = Column(JSON)


class ApprovalRecord(Base):
    __tablename__ = "approval_records"

    id           = Column(String, primary_key=True)
    run_id       = Column(String, index=True, nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    decision     = Column(String, nullable=False)   # "approved" | "rejected"
    reviewer_note = Column(Text)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
