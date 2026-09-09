# Book Summarizer & Query Agent

RAG-based agentic book summarizer + Q&A over uploaded PDFs/DOCX/TXT.

- **Backend**: FastAPI + LangGraph + pgvector (Postgres) + Groq (chat) + Google Gemini (embeddings)
- **Frontend**: Next.js
- **Docs**: [`docs/architecture.md`](docs/architecture.md), [`docs/api.md`](docs/api.md)

## Quick start (local, Docker)

```bash
cp backend/.env.example backend/.env
# Set your credentials — minimum required for default providers:
#   CHAT_PROVIDER=groq          → fill in GROQ_API_KEY, set GROQ_CHAT_MODEL=groq/compound-mini
#   EMBEDDING_PROVIDER=gemini   → fill in GEMINI_API_KEY
# Azure OpenAI (recommended for production): set CHAT_PROVIDER=azure_openai,
#   EMBEDDING_PROVIDER=azure_openai and fill in AZURE_OPENAI_* vars.

docker compose up --build
# frontend: http://localhost:3000
# backend docs: http://localhost:8000/docs
```

## Quick start (no Docker)

```bash
# 1. Postgres with pgvector (or use docker for just the db):
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=bookagent ankane/pgvector

# 2. Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in credentials
uvicorn app.main:app --reload

# 3. Frontend
cd ../frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

## Suggested 5-day execution plan

- **Day 1**: Provision Azure OpenAI (or Bedrock) + Postgres w/ pgvector. Get this scaffold running end-to-end locally with a small test PDF. Verify upload → chunk → embed → summarize → Q&A loop works.
- **Day 2**: Harden ingestion for real 500+ page books (timeouts, batching, retries on LLM calls, progress status). Add Alembic migration for the schema instead of `create_all`.
- **Day 3**: Polish frontend (loading states, error states, delete button, simple login/signup form wired to `/auth/signup` and `/auth/login`). Add LangSmith tracing env vars.
- **Day 4**: Deploy: Azure Container Apps/App Service for backend+frontend, Azure Database for PostgreSQL Flexible Server (enable `vector` extension), Azure OpenAI resource. Point `NEXT_PUBLIC_API_BASE` at the deployed backend URL.
- **Day 5**: End-to-end test with multiple large books and two different browsers/users to confirm history isolation, write final README notes/screenshots, record a short demo, submit GitHub link + live URL.

## Repo layout

```
backend/
  app/
    api/        - FastAPI routers (auth, books, chat)
    core/       - config, db session, JWT auth, logging
    models/     - SQLAlchemy models + Pydantic schemas
    services/   - document extraction, chunking, RAG, LLM client, ingestion
    agents/     - LangGraph graphs (summarize, RAG Q&A)
frontend/
  app/          - Next.js pages (upload/history, per-book chat)
  lib/api.js    - API client + anonymous session handling
docs/
  architecture.md
  api.md
docker-compose.yml
```
