# Architecture — Book Summarizer & Query Agent

## 1. High-level components

```
┌────────────┐      HTTPS/JSON       ┌────────────────┐      SQL/pgvector      ┌──────────────┐
│  Next.js   │  ───────────────────▶ │   FastAPI       │  ────────────────────▶ │  PostgreSQL  │
│  Frontend  │ ◀───────────────────  │   Backend       │ ◀──────────────────── │  + pgvector  │
└────────────┘                       └───────┬─────────┘                       └──────────────┘
                                              │
                                              ▼
                                   ┌────────────────────┐
                                   │  LangGraph Agents   │
                                   │  - Summarize graph  │
                                   │  - RAG Q&A graph    │
                                   └───────┬────────────┘
                                           │
                             ┌─────────────┴──────────────┐
                             ▼                            ▼
                   ┌──────────────────┐       ┌──────────────────────┐
                   │  Groq (chat)     │       │  Google Gemini       │
                   │  compound-mini   │       │  (embeddings only)   │
                   │  chat/completions│       │  gemini-embedding-001│
                   └──────────────────┘       └──────────────────────┘
```

- **Frontend**: Next.js (React). Single-page-ish app: upload page + history list + per-book chat page.
- **Backend**: FastAPI, split into `api/` (routers), `core/` (config, db, security, logging), `models/` (SQLAlchemy + Pydantic), `services/` (document processing, RAG, LLM client, ingestion), `agents/` (LangGraph graphs). This separation is what makes it easy to add new agents, new file types, or a new LLM provider without touching unrelated code (modularity requirement).
- **Database**: PostgreSQL with the `pgvector` extension for similarity search, in the same instance as normal relational tables (users, books, chunks, chat_messages) — no separate vector DB needed, keeping ops simple.
- **LLM providers**: Independently configurable for chat and embeddings via two separate env vars (`CHAT_PROVIDER` / `EMBEDDING_PROVIDER`). All provider logic is isolated in `LLMClient` (`services/llm_client.py`) — switching providers is a config-only change.

## 2. LLM Provider Design

### Production configuration

| Role | Provider | Model | Notes |
|---|---|---|---|
| Chat / completions | Groq | `groq/compound-mini` | Map-reduce summarizer + RAG Q&A generation |
| Embeddings | Google Gemini | `gemini-embedding-001` | Chunk + query embeddings, 768-dim output |

### Why this split (development history)

During development this project evaluated several providers before settling on the current split:

- **Azure OpenAI** was the original target (planned default in the scaffold). Azure's access-approval process did not complete within the project timeline, so it could not be used as the primary provider, though the `LLMClient` fully supports it and it remains the recommended production target for a paid deployment.
- **GitHub Models** (Azure inference proxy): tested and working for both chat and embeddings; hit per-day rate limits too quickly for iterative multi-call agentic pipeline testing.
- **Gemini alone** (chat + embeddings): the free tier's **20 requests-per-day chat cap** was exhausted in a single testing session; the cap makes it unsuitable as the sole chat provider during development.
- **Groq alone** (chat + embeddings): Groq's free tier provides generous chat RPM/RPD limits, but its embedding API is not available on all plans.
- **Provider split solution**: Groq's generous chat limits and Gemini's generous embedding limits are complementary. Using Groq for chat and Gemini for embeddings avoided hitting either limit during iterative pipeline testing. A paid Azure OpenAI deployment would consolidate both back to a single provider with no code changes.

### Configuration

```
CHAT_PROVIDER=groq            # groq | azure_openai | github_models | gemini | aws_bedrock | openai
EMBEDDING_PROVIDER=gemini     # gemini | azure_openai | github_models | openai
```

`chat_provider` and `embedding_provider` are **independent** settings — they do not have to match. `LLMClient` initialises a separate SDK client for each (`_chat_client` / `_embed_client`) so both providers run simultaneously.

## 3. Document ingestion & storage strategy

Why chunking, not "one vector per book": embedding models have finite context, and a single vector for a 500-page book would be a very lossy compression — retrieval quality collapses because a query about "chapter 12" and a query about "chapter 2" produce nearly identical similarity scores against a single whole-book vector.

Chosen strategy:
1. Extract raw text from PDF/DOCX/TXT (`pypdf` / `python-docx`).
2. Split into **token-based sliding-window chunks** (~800 tokens, 120-token overlap) using `tiktoken`. Overlap prevents losing context that straddles a chunk boundary.
3. Embed each chunk (`gemini-embedding-001`, 768-dim output via `output_dimensionality=768`) and store in the `chunks` table with a `vector` column, plus its `ordinal` (position in the book) so neighboring chunks can be pulled for extra context later.
4. The **100-word summary** is produced separately via a **map-reduce LangGraph workflow** (not sent as one giant prompt, and not one-shot):
   - **Map**: batches of ~6 chunks are compressed into short notes in parallel-style LLM calls (max_tokens=300).
   - **Reduce**: all notes are combined into a single ~100-word summary (max_tokens=600).
   - **Verify**: a lightweight word-count/faithfulness check triggers a rewrite loop (max 2 attempts) if the draft is off-target — this is the "agentic" loop the assignment asks for, versus one-shot prompting.

## 4. RAG Q&A flow (LangGraph)

`retrieve → grade → generate` (with a `not_relevant` branch):
1. **Retrieve**: embed the user's question (Gemini, 768-dim), run a cosine-distance query (`<=>` operator) against that book's chunks only, top-k=5.
2. **Grade**: cheap similarity-floor heuristic decides if retrieved context is usable (swappable for an LLM-based grader later — the graph node is isolated for that).
3. **Generate**: Groq answers **using only the retrieved chunks** (explicit "don't hallucinate outside context" instruction) plus the last 4 turns of conversation history (each truncated to 300 chars to stay within payload limits). The answer is written as clean prose; chunk citations are grouped into a single `Sources: [chunk N, chunk M]` line at the end.
4. Every Q&A turn is persisted to `chat_messages`, which is what powers both "continue this conversation" and the "history of extracted books" view.

## 5. Auth & multi-tenancy

- Every browser tab/window with no session yet calls `POST /auth/anonymous` once, gets a JWT tied to a freshly created anonymous `User` row, and stores it in `sessionStorage` (cleared per browser session/tab → "each session unique to the browser window").
- Optional real accounts: `POST /auth/signup` / `POST /auth/login` issue the same JWT shape, so all downstream code (`get_current_user`) doesn't need to know which mode was used.
- Every book/chat query is filtered by `owner_id == current_user.id`; requests for another user's book return `404` (not `403`, to avoid leaking existence) and unauthenticated requests get `401`.

## 6. Observability

- `structlog` JSON logs for every request (method, path, status, latency, request-id) via a Starlette middleware — logs can be shipped to Azure Monitor / CloudWatch / any log sink as-is.
- LLM calls are logged (provider, message count, payload size) and instrumented for LangSmith tracing when `LANGSMITH_TRACING=true` (LangGraph integrates with LangSmith with zero code changes beyond env vars) for step-by-step agent trace visibility.
- All exceptions are caught by global handlers and converted to structured `500`s with a request id, never a raw stack trace to the client.

## 7. Error handling & HTTP status codes

| Situation | Code |
|---|---|
| Book uploaded, processing started | 202 Accepted |
| Successful GET | 200 OK |
| Delete book | 204 No Content |
| Missing/invalid auth token | 401 Unauthorized |
| Book not found / not owned by user | 404 Not Found |
| Unsupported file extension | 415 Unsupported Media Type |
| File too large | 413 Payload Too Large |
| Asking a question before processing finishes | 409 Conflict |
| Bad request body | 422 Unprocessable Entity |
| Duplicate signup email | 409 Conflict |
| Unhandled server error | 500 Internal Server Error |

## 8. Known Constraints & Development Notes

### Free-tier rate limits

The current provider split was chosen specifically to work within free-tier constraints:

| Provider | Constraint | Impact |
|---|---|---|
| Groq free tier | Requests-per-minute / requests-per-day for chat | Ingestion of large books (many map calls) hits RPM; tenacity retry+backoff handles this automatically |
| Gemini free tier | 20 requests-per-day for chat; generous embedding RPD | Gemini used **only** for embeddings; chat moved to Groq |
| Gemini embeddings | `output_dimensionality=768` set explicitly | Reduces vector size from 3072 → 768 dims; lowers storage and distance-computation cost with negligible quality impact for book-length text |

### Retry & backoff

`LLMClient` uses `tenacity` for automatic retry: up to 10 attempts for chat calls, 5 for embedding calls. On a `429 Too Many Requests` response the client parses the retry-after delay from the error message and waits accordingly before retrying.

### Payload size management

Groq enforces a per-request token limit. The `generate()` node manages this by:
- Capping retrieved chunks to `top_k=5` (was 8).
- Limiting conversation history to the last 4 messages (was 6).
- Truncating each historical message to 300 characters before including it in the prompt.

A `413 request_too_large` error from Groq is caught and returns a user-friendly message instead of crashing.

### pgvector on Windows Postgres 17

The `pgvector` extension is not available as a pre-built binary for PostgreSQL 17 on Windows. The local dev setup uses a fallback: a custom `vector DOMAIN AS float8[]` type plus a PL/pgSQL `vector_cosine_distance()` function and `<=>` operator defined at application startup. Production deployments on Azure Database for PostgreSQL Flexible Server (or any Linux Postgres) should enable the real `pgvector` extension instead.

### Production path

A paid Azure OpenAI deployment would:
- Consolidate chat + embeddings back to a single provider (`CHAT_PROVIDER=azure_openai`, `EMBEDDING_PROVIDER=azure_openai`).
- Remove free-tier rate-limit concerns entirely.
- No code changes required — only `.env` updates.

## 9. Deployment (Azure)

- **Azure Database for PostgreSQL Flexible Server** with the `vector` extension enabled (allow-list it in server parameters).
- **Azure OpenAI** for chat + embeddings (set `CHAT_PROVIDER=azure_openai`, `EMBEDDING_PROVIDER=azure_openai`, fill in `AZURE_OPENAI_*` vars).
- **Azure App Service (Linux, container)** or **Azure Container Apps** for both the FastAPI backend and Next.js frontend (two containers from `docker-compose.yml`).
- Secrets (DB URL, API keys) go into **Azure Key Vault** / App Service application settings, never committed.

## 10. Future extensibility (by design)

- New file types → add a branch in `document_processing.extract_text`.
- New LLM provider → add a branch in `LLMClient` (`services/llm_client.py`); no changes to agents or routers.
- New agent behavior (e.g., a "compare two books" agent) → new LangGraph graph in `agents/`, new router in `api/`, no changes to existing graphs/routers.
- Swap in-process background task for Celery/RQ workers for real scale — `ingestion.process_book` is already a standalone function with no FastAPI dependency, so it drops into a worker unchanged.
