# Book Summarizer & Query Agent

An end-to-end, RAG-based agentic document processing system. It extracts, chunks, embeds, generates map-reduce summaries (~100 words), and answers questions over uploaded PDF, DOCX, and TXT books.

---

## 🛠️ Technology Stack

### Backend & Core Infrastructure
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+) + [Uvicorn](https://www.uvicorn.org/) ASGI server
- **Database & Vector Search**: [PostgreSQL](https://www.postgresql.org/) with the [`pgvector`](https://github.com/pgvector/pgvector) extension (using `pgvector-python` & SQLAlchemy 2.0 ORM, with native SQL fallback for environments without `pgvector` pre-installed)
- **Agent Workflows**: [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph` map-reduce summarizer agent & multi-turn RAG Q&A graph)
- **Data Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/latest/) + `pydantic-settings`

### AI Models & Multi-Provider LLM Architecture
- **Unified Client**: Modular `LLMClient` supporting independent chat and embedding provider configuration:
  - **Groq**: Primary high-speed chat completion using `openai/gpt-oss-20b` (with automatic failover to Gemini Flash model rotation)
  - **Google Gemini**: Multi-tier chat fallback rotating across 5 Flash model variants (`gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash`, `gemini-3.7-flash`) & `gemini-embedding-001` (768-dim embeddings via `google-genai` SDK)
  - **Azure OpenAI**: Native support for `gpt-4o-mini` and `text-embedding-3-small`
  - **GitHub Models** & **AWS Bedrock** (`boto3`)
  - **Local Embeddings**: `SentenceTransformer` (`all-MiniLM-L6-v2`)
- **Resilience & Retries**: Exponential backoff, 503/high-demand error handling, and rate-limit mitigation via [`tenacity`](https://github.com/jd/tenacity)

### Document Processing & Ingestion
- **Parsers**: `pypdf` (PDF text extraction), `python-docx` (DOCX extraction), plain UTF-8 reader (TXT)
- **Chunking**: Token-aware sliding window chunker powered by `tiktoken` (`cl100k_base` encoding, 800 tokens per chunk with 120 token overlap)

### Authentication & Multi-Tenancy
- **Security**: JWT authentication (`python-jose`, `passlib` with `bcrypt`)
- **Isolation**: Anonymous window-scoped sessions (stored in browser `sessionStorage` via `POST /auth/anonymous`) and user account authentication (`POST /auth/signup`, `POST /auth/login`). Data access is strictly isolated by `owner_id`.

### Observability & Logging
- **Structured Logging**: `structlog` JSON logs with request correlation IDs via custom Starlette middleware (`RequestContextMiddleware`) and isolated background task logging (`path="background_ingestion"`)
- **Tracing**: Native integration with [LangSmith](https://www.langchain.com/langsmith) tracing (`LANGSMITH_TRACING=true`)

### Frontend
- **Framework**: [Next.js 14](https://nextjs.org/) (App Router, React 18)
- **API Integration**: Lightweight custom API wrapper with anonymous session auto-handshake
- **UI**: Modern responsive interface with drag-and-drop file upload, real-time polling status, summary cards, and multi-turn chat window

### DevOps & Deployment
- **Containerization**: Docker & `docker-compose` (`ankane/pgvector`, backend, frontend) with optimized `.dockerignore` files
- **PaaS Deployment**: Ready for deployment on [Render](https://render.com/) (`render.yaml`) or Azure Container Apps / Azure App Service

---

## 🚀 Quick Start

### Option A: Local with Docker (Recommended)

1. **Clone the repository and prepare configuration**:
   ```bash
   cp backend/.env.example backend/.env
   ```
2. **Set API Keys in `backend/.env`**:
   - For default **Gemini** provider: fill in `GEMINI_API_KEY`
   - For **Groq** provider: set `CHAT_PROVIDER=groq` and fill in `GROQ_API_KEY`
3. **Launch all services**:
   ```bash
   docker compose up --build
   ```
4. **Access the application**:
   - **Frontend UI**: [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Local without Docker

1. **Start Postgres with pgvector**:
   ```bash
   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=bookagent ankane/pgvector
   ```

2. **Setup and run Backend**:
   ```bash
   cd backend
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   cp .env.example .env   # Configure your GEMINI_API_KEY / GROQ_API_KEY
   uvicorn app.main:app --reload
   ```

3. **Setup and run Frontend**:
   ```bash
   cd ../frontend
   npm install
   NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
   ```

---

## 🔑 Environment Variables Configuration

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://postgres:postgres@localhost:5432/bookagent` |
| `JWT_SECRET` | Secret key for signing JWT tokens | `change-me-in-prod` |
| `CHAT_PROVIDER` | LLM provider for chat completions (`gemini`, `groq`, `azure_openai`, `github_models`, `aws_bedrock`) | `gemini` |
| `EMBEDDING_PROVIDER` | Provider for vector embeddings (`gemini`, `azure_openai`, `github_models`, `aws_bedrock`, `local`) | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API Key | `""` |
| `GEMINI_CHAT_MODEL` | Primary Gemini chat model | `gemini-3.6-flash` |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model | `gemini-embedding-001` |
| `GROQ_API_KEY` | Groq API Key | `""` |
| `GROQ_CHAT_MODEL` | Primary Groq chat model | `openai/gpt-oss-20b` |
| `EMBEDDING_DIM` | Vector dimensionality | `768` |
| `LANGSMITH_TRACING` | Enable agent tracing in LangSmith | `false` |

---

## 📁 Repository Layout

```
book-agent/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph agents (map-reduce summarizer & RAG Q&A graph)
│   │   ├── api/             # FastAPI endpoint routers (auth, books, chat)
│   │   ├── core/            # App configuration, DB session, JWT security, structured logging
│   │   ├── models/          # SQLAlchemy ORM database models & Pydantic request/response schemas
│   │   ├── services/        # Document extraction, token chunking, RAG retrieval, LLMClient
│   │   └── main.py          # FastAPI app entrypoint, middleware, exception handlers
│   ├── .dockerignore        # Docker build context exclusions for backend
│   ├── Dockerfile           # Backend container image build
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── app/                 # Next.js App Router pages (library home & per-book Q&A chat page)
│   ├── lib/api.js           # API client with anonymous session handling
│   ├── .dockerignore        # Docker build context exclusions for frontend
│   ├── Dockerfile           # Frontend container image build
│   └── package.json         # Node.js dependencies
├── docs/
│   ├── architecture.md      # In-depth architectural design, provider trade-offs & RAG flow
│   └── api.md               # API endpoint specification & JSON schemas
├── docker-compose.yml       # Multi-container orchestration (DB, Backend, Frontend)
└── render.yaml              # Render PaaS deployment configuration
```

---

## 📖 Documentation Links

- [Architecture Guide](docs/architecture.md): Full breakdown of multi-tier provider failover, chunking decisions, map-reduce summarization graph, and multi-tenancy model.
- [API Reference](docs/api.md): Endpoint list, authentication headers, request payload formats, and status codes.
