# AI Sequence Generator

Production-grade agentic AI system for generating personalised outbound email sequences. Built with **LangGraph**, **multi-provider LLM orchestration**, **live website scraping → RAG**, **human-in-the-loop approval**, and **full observability**.

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
 + Company  ──▶│ Scraper │──▶│ Planner  │──▶│ Research │──▶│  Writer  │──▶ Reviewer ──▶ Human
   URL         │ (httpx) │    │ Agent    │    │ Agent    │    │  Agent   │   ▲   │       Approval
               └─────────┘    └──────────┘    │ (RAG)    │    └──────────┘   │   │
                    │                         └──────────┘                   │   │
                    ▼                                                        │   │
               ┌─────────┐                                         score < 7│   │
               │ Qdrant  │◀── embed chunks ──────────────────── revision ◀──┘   │
               │ (Vector │                                      loop (max 3)    │
               │  Store) │                                                      ▼
               └─────────┘                                                   __end__
```

## Key Features

- **Live Website Scraping → RAG**: Paste any company URL — the system scrapes the site, chunks text into ~300-word pieces, embeds via Gemini, and stores in Qdrant. RAG now works with **any company in the world**, not just seeded data.
- **4 Specialized AI Agents** coordinated by LangGraph (not a single prompt)
- **Multi-provider LLM** with circuit breaker (Groq + Gemini, automatic failover)
- **Self-correction loop** (Reviewer sends bad emails back to Writer, up to 3 cycles)
- **Human-in-the-loop approval** (nothing auto-publishes)
- **Per-agent observability** (tokens, latency, model per step)
- **Real-time SSE streaming** (user sees each agent working live)

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Add your GOOGLE_API_KEY and/or GROQ_API_KEY

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Start Qdrant (optional — RAG degrades gracefully without it)
docker-compose up -d qdrant

# 4. Run
uvicorn app.main:app --reload --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

### Provider Setup (Free Tier)

| Provider | Free Tier | How to Get Key |
|----------|-----------|----------------|
| **Groq** (recommended) | 30 RPM, 14,400 RPD | [console.groq.com](https://console.groq.com) |
| **Gemini** | 10 RPM, 500 RPD | [aistudio.google.com](https://aistudio.google.com) |

The system works with either or both. If both are configured, Groq is preferred (faster) with Gemini as automatic fallback.

## How It Works — Step by Step

### 1. Website Scraping (`app/rag/scraper.py`)
When the user provides a company URL:
- Fetches homepage + up to 4 internal links (concurrent via httpx)
- Strips non-content HTML (scripts, nav, footer) with BeautifulSoup
- Chunks clean text into ~300-word pieces
- Embeds each chunk via Gemini (`gemini-embedding-2`, 3072-dim)
- Upserts to Qdrant tagged with `run_id` for per-run scoping
- URL validation with SSRF protection (blocks localhost/private IPs)
- Graceful degradation: if scraping fails, the workflow continues with LLM knowledge

### 2. Planner Agent (`app/agents/planner.py`)
Two-tier goal validation:
- **Tier 1 — Heuristics** (zero cost, instant): rejects harmful/personal/chatbot inputs
- **Tier 2 — LLM** (only for ambiguous goals): fail-open on API errors
- Then produces a structured campaign strategy (tone, CTA, talking points)

### 3. Research Agent (`app/agents/research.py`)
- Searches Qdrant filtered by `run_id` — only sees this run's scraped content
- Retrieves top-6 semantically similar chunks by cosine similarity
- LLM synthesizes a research brief grounded in real scraped data
- Falls back to LLM-generated industry context if Qdrant is empty

### 4. Writer Agent (`app/agents/writer.py`)
- Takes plan + research context + optional revision feedback
- Generates personalised cold email + follow-up
- Uses `[First Name]`, `[Company Name]` placeholders — never hallucinates names

### 5. Reviewer Agent (`app/agents/reviewer.py`)
- Scores on 5 weighted dimensions (personalization, clarity, relevance, deliverability, structure)
- **Verdict computed in Python** — never trusts LLM's self-assessment for control flow
- Score < 7.0 → routes back to Writer (up to 3 revision cycles)

### 6. Human Approval
- Results saved with status `"pending"`
- Human reviews, approves or rejects with notes
- Full audit trail in `approval_records` table

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

### Request Schema

```json
{
  "goal": "Book meetings with CTOs",
  "persona": "CTO",
  "company": "Stripe",
  "company_url": "https://stripe.com"   // optional — enables live RAG
}
```

### SSE Stream Events

```
POST /api/generate/stream
→ data: {"type":"agent_start","agent":"scraper","label":"Website Scraper"}
→ data: {"type":"agent_complete","agent":"scraper","chunks":12}
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
│   ├── llm_provider.py     # Abstract LLMProvider interface (Strategy Pattern)
│   ├── gemini_provider.py  # Google Gemini with retry + fallback
│   ├── groq_provider.py    # Groq LPU inference (LLaMA, Mixtral)
│   ├── openai_provider.py  # OpenAI (pluggable, not active by default)
│   └── provider_router.py  # SmartRouter: circuit breaker + health tracking
├── rag/
│   ├── scraper.py          # Live website scraping → chunking → embedding → Qdrant
│   ├── embeddings.py       # Gemini embedding-2 wrapper (3072-dim)
│   ├── retrieval.py        # Qdrant semantic search with run_id filtering
│   ├── qdrant_client.py    # Client with graceful degradation
│   └── seed_knowledge.py   # Sample knowledge base seeder (fallback)
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
| **RAG** | Live website scraping | `scraper.py` — httpx + BeautifulSoup → chunk → embed → Qdrant |
| **RAG** | Per-run scoping | Qdrant filter by `run_id` — each run only sees its own scraped data |
| **RAG** | Semantic search | Qdrant vector store with graceful degradation |
| **LLM Orchestration** | Multi-provider routing | `SmartRouter` with Groq + Gemini |
| **Resilience** | Circuit breaker | Opens after 3 failures, half-opens after 60s cooldown |
| **Resilience** | Per-provider retry | Exponential backoff via `tenacity` |
| **Resilience** | Automatic fallback | Cross-provider + intra-provider model fallback |
| **Resilience** | SSRF protection | URL validation blocks localhost/private IPs |
| **Streaming** | Real-time pipeline | SSE via `StreamingResponse` + `asyncio.Queue` |
| **Observability** | Per-agent trace | Model, tokens, latency attributed per agent step |
| **Observability** | Provider health | Live success rate, avg latency, circuit status |
| **Observability** | Token tracking | Prompt + completion tokens per run |
| **Observability** | Prompt versioning | `<agent>_v<N>.txt` with version tag per run |
| **Quality** | Score-based routing | Conditional edges in LangGraph (reviewer → writer loop) |
| **Quality** | Goal validation | Two-tier: heuristic pre-filter + LLM validation |
| **Human-in-Loop** | Approval UI | Review, approve/reject with audit trail |
| **Logging** | Structured logs | JSON in production, rich formatting in dev |

## Error Handling — Every Failure Scenario

| What Can Fail | What Happens | Code Location |
|---------------|-------------|---------------|
| Invalid company URL | URL validation rejects, scraper skipped, workflow continues | `scraper.py:validate_url()` |
| Company site unreachable | Scraper returns 0 chunks, UI shows warning, workflow continues | `scraper.py:_fetch_page()` |
| Site blocks bots | 403/timeout, scraper returns 0 chunks, RAG falls back to LLM knowledge | `scraper.py:_fetch_page()` |
| Invalid goal ("buy milk") | Tier 1 heuristics reject instantly, HTTP 400 | `planner.py:check_goal_heuristics()` |
| Primary LLM rate-limited | SmartRouter tries next provider | `provider_router.py:complete()` |
| All LLM providers down | Circuit breaker open, clear error to user | `provider_router.py:complete()` |
| Qdrant down | Research Agent uses LLM-generated context | `retrieval.py:retrieve_context()` |
| Embedding fails | Returns `[]`, RAG skipped | `embeddings.py:embed_text()` |
| LLM returns invalid JSON | Agent uses hardcoded fallback dict | Every agent's `except JSONDecodeError` |
| Low-quality email | Reviewer routes back to Writer (up to 3 times) | `workflow.py:route_after_review()` |

## Design Patterns

| Pattern | Where | Backend Equivalent |
|---------|-------|-------------------|
| **Strategy Pattern** | `LLMProvider` ABC + Groq/Gemini implementations | JDBC drivers, payment gateway interfaces |
| **Circuit Breaker** | `SmartRouter` with `ProviderHealth` tracking | Netflix Hystrix, Resilience4j |
| **Singleton** | `SmartRouter` instance shared across requests | Connection pools, cache instances |
| **State Machine** | LangGraph `StateGraph` with conditional edges | Workflow engines (Temporal, Step Functions) |
| **Observer/Callback** | `MetricsCollector.emit_progress()` → SSE stream | Event listeners, pub/sub |
| **Graceful Degradation** | RAG fallback, JSON parse fallback, provider failover | Circuit breakers, fallback responses |
| **Atomic Write** | Metrics flushed in single DB commit | Database transactions |

## Tech Stack

| Technology | Why |
|-----------|-----|
| **FastAPI** | Async-native, auto-generates OpenAPI docs, Pydantic validation |
| **LangGraph** | StateGraph with typed state + conditional edges — better than chains for loops |
| **Groq** | Free tier, sub-200ms inference on custom LPU hardware |
| **Gemini** | Free tier, reliable, good at structured JSON output |
| **Qdrant** | Open-source vector DB, Docker-ready, clean async Python SDK |
| **httpx + BeautifulSoup** | Async HTTP + HTML parsing for live website scraping |
| **SQLite + aiosqlite** | Zero-config, async, easy to swap to PostgreSQL |
| **SSE** | Simpler than WebSockets for one-way server→client push |

## Docker

```bash
cp .env.example .env   # fill API keys
docker-compose up --build
```
