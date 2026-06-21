# AI Sequence Generator

Production-grade agentic AI system for generating personalised outbound email sequences. Built with **LangGraph**, **multi-provider LLM orchestration**, **RAG**, **human-in-the-loop approval**, and **full observability**.

> See [ARCHITECTURE.md](./ARCHITECTURE.md) for an in-depth technical deep-dive covering system design, concepts, and implementation details.

## System Architecture

```
                         ┌─────────────────────┐
                         │   SmartRouter        │
                         │  (Circuit Breaker)   │
                         ├──────────┬──────────┤
                         │  Groq    │  Gemini  │
                         │  (LPU)   │  (Cloud) │
                         └────┬─────┴────┬─────┘
                              │          │
                    ┌─────────▼──────────▼─────────┐
                    │     LangGraph StateGraph      │
                    │                               │
User Input ──▶ ┌────┴────┐    ┌──────────┐    ┌────┴────┐    ┌──────────┐
               │ Planner │──▶│ Research  │──▶│  Writer  │──▶│ Reviewer │──▶ Human
               │ Agent   │    │ Agent    │    │  Agent   │    │  Agent   │   Approval
               └─────────┘    │ (RAG)    │    └────┬─────┘    └──┬───────┘
                              └──────────┘         │             │
                                              ◀────┘   score < 7│
                                              revision loop ◀────┘
                                              (max 3 cycles)
```

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Add your GOOGLE_API_KEY and/or GROQ_API_KEY

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run
uvicorn app.main:app --reload --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

### Provider Setup (Free Tier)

| Provider | Free Tier | How to Get Key |
|----------|-----------|----------------|
| **Groq** (recommended) | 30 RPM, 14,400 RPD | [console.groq.com](https://console.groq.com) |
| **Gemini** | 10 RPM, 500 RPD | [aistudio.google.com](https://aistudio.google.com) |

The system works with either or both. If both are configured, Groq is preferred (faster) with Gemini as automatic fallback.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Run workflow (sync) |
| `POST` | `/api/generate/stream` | Run workflow (SSE streaming) |
| `GET`  | `/api/runs` | List recent runs |
| `GET`  | `/api/runs/{run_id}` | Run details + agent trace |
| `POST` | `/api/approve/{run_id}` | Approve a run |
| `POST` | `/api/reject/{run_id}` | Reject a run |
| `GET`  | `/api/health/providers` | Provider health + circuit breaker status |

### SSE Stream Events

```
POST /api/generate/stream
Content-Type: application/json
{"goal": "Book meetings with CTOs", "persona": "CTO", "company": "Acme Inc"}

→ data: {"type":"agent_start","agent":"planner","label":"Planner Agent"}
→ data: {"type":"agent_complete","agent":"planner","model":"llama-3.3-70b-versatile"}
→ data: {"type":"agent_start","agent":"research","label":"Research Agent (RAG)"}
→ data: {"type":"agent_complete","agent":"research","model":"llama-3.3-70b-versatile"}
→ data: {"type":"agent_start","agent":"writer","label":"Writer Agent"}
→ data: {"type":"agent_complete","agent":"writer","model":"llama-3.3-70b-versatile"}
→ data: {"type":"agent_start","agent":"reviewer","label":"Reviewer Agent"}
→ data: {"type":"agent_complete","agent":"reviewer","model":"llama-3.3-70b-versatile","score":8.5}
→ data: {"type":"complete","run_id":"...","score":8.5,"total_tokens":2100}
```

## Project Structure

```
app/
├── agents/
│   ├── planner.py          # Campaign strategy + goal validation (heuristic + LLM)
│   ├── research.py         # RAG retrieval + research brief synthesis
│   ├── writer.py           # Email + follow-up generation with revision support
│   └── reviewer.py         # Multi-dimensional scoring + deterministic routing
├── graph/
│   └── workflow.py         # LangGraph StateGraph with progress event emission
├── providers/
│   ├── llm_provider.py     # Abstract LLMProvider interface (2 methods)
│   ├── gemini_provider.py  # Google Gemini with retry + fallback
│   ├── groq_provider.py    # Groq LPU inference (LLaMA, Mixtral)
│   ├── openai_provider.py  # OpenAI (pluggable, not active by default)
│   └── provider_router.py  # SmartRouter: circuit breaker + health tracking
├── rag/
│   ├── embeddings.py       # Embedding model wrapper
│   ├── retrieval.py        # Qdrant semantic search
│   ├── qdrant_client.py    # Client with graceful degradation
│   └── seed_knowledge.py   # Sample knowledge base seeder
├── observability/
│   ├── metrics.py          # Per-run + per-agent token/latency collector
│   ├── logger.py           # Structured logging (JSON in prod, rich in dev)
│   └── prompt_loader.py    # Versioned prompt loader with LRU cache
├── prompts/                # Versioned prompt templates (planner_v1.txt, etc.)
├── api/
│   └── routes.py           # FastAPI routes + SSE streaming + health endpoint
├── db/
│   └── models.py           # SQLAlchemy async models (execution_logs, approvals)
├── templates/              # Jinja2 templates (dashboard, generate, review, logs)
├── config.py               # Pydantic settings (multi-provider config)
└── main.py                 # FastAPI app entry point
```

## Production Features

| Category | Feature | Implementation |
|----------|---------|---------------|
| **LLM Orchestration** | Multi-provider routing | `SmartRouter` with Groq + Gemini |
| **Resilience** | Circuit breaker | Opens after 3 failures, half-opens after 60s cooldown |
| **Resilience** | Per-provider retry | Exponential backoff via `tenacity` |
| **Resilience** | Automatic fallback | Cross-provider + intra-provider model fallback |
| **Streaming** | Real-time pipeline | SSE via `StreamingResponse` + `asyncio.Queue` |
| **Observability** | Per-agent trace | Model, tokens, latency attributed per agent step |
| **Observability** | Provider health | Live success rate, avg latency, circuit status |
| **Observability** | Token tracking | Prompt + completion tokens per run |
| **Observability** | Prompt versioning | `<agent>_v<N>.txt` with version tag per run |
| **RAG** | Semantic search | Qdrant vector store with graceful degradation |
| **Quality** | Score-based routing | Conditional edges in LangGraph (reviewer → writer loop) |
| **Quality** | Goal validation | Two-tier: heuristic pre-filter + LLM validation |
| **Human-in-Loop** | Approval UI | Review, approve/reject with audit trail |
| **Logging** | Structured logs | JSON in production, rich formatting in dev |

## Docker

```bash
cp .env.example .env   # fill API keys
docker-compose up --build
```
