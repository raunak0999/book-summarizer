# API Documentation

Base URL: `http://localhost:8000` (or your deployed backend URL).
Interactive OpenAPI docs are auto-served at `/docs` (Swagger UI) and `/redoc`.

All endpoints except `/health` and `/auth/*` require:
`Authorization: Bearer <jwt>`

---

## Auth

### `POST /auth/anonymous`
Creates (or is called once per browser session to obtain) an anonymous user + JWT. This is what the frontend calls automatically on first load.
- **Body**: none
- **200**: `{ "access_token": "...", "token_type": "bearer" }`

### `POST /auth/signup`
Create a real account.
- **Body**: `{ "email": "a@b.com", "password": "..." }`
- **201**: token, same shape as above
- **409**: email already registered

### `POST /auth/login`
- **Body**: `{ "email": "a@b.com", "password": "..." }`
- **200**: token
- **401**: invalid credentials

---

## Books

### `POST /books`
Upload a book (PDF/DOCX/TXT). Kicks off async processing (extraction → chunk → embed → map-reduce summarize).
- **Body**: multipart/form-data, field `file`
- **202**: `BookOut` with `status: "processing"`
- **415**: unsupported file extension
- **413**: file exceeds size limit (default 60MB)

### `GET /books`
List the current user's book history (most recent first). This is the "history of extracted books" feature — strictly scoped to `owner_id`.
- **200**: `BookOut[]`

### `GET /books/{book_id}`
Poll a single book's status/summary (frontend polls this while `status == "processing"`).
- **200**: `BookOut`
- **404**: not found / not owned by caller

### `DELETE /books/{book_id}`
Deletes a book and all its chunks/chat history (cascade).
- **204**: no content
- **404**: not found / not owned by caller

`BookOut` shape:
```json
{
  "id": "uuid",
  "filename": "moby_dick.pdf",
  "status": "processing | ready | failed",
  "page_count": 512,
  "summary_100w": "…exactly ~100 words…",
  "error_message": null,
  "created_at": "2026-09-06T10:00:00Z"
}
```

---

## Chat (RAG Q&A)

### `POST /books/{book_id}/chat`
Ask a question about a specific book. Runs the LangGraph RAG agent (retrieve → grade → generate) and persists both the question and answer.
- **Body**: `{ "question": "Who is the antagonist?" }`
- **200**: `AnswerResponse` (see shape below)
- **404**: book not found / not owned by caller
- **409**: book still processing (`status != "ready"`)

`AnswerResponse` shape:
```json
{
  "answer": "Elric Combe was the keeper of the Amberlyn Light for thirty years...\n\nSources: [chunk 3, chunk 4, chunk 12]",
  "sources": [3, 4, 12]
}
```

Two citation mechanisms are present for different consumers:
- **`sources` field** — machine-readable array of integer chunk ordinals. Use this to look up raw chunk content, build "jump to source" UI, or do programmatic grounding checks.
- **`Sources:` line in `answer` text** — human-readable citation appended at the end of the prose by the language model. The answer body itself is written as clean flowing prose (no inline `[chunk N]` interruptions mid-sentence); all citations are grouped on the final line. This makes the answer comfortable to read while still being auditable.

Retrieval uses cosine similarity against the book's embedded chunks (top-k = 5). Conversation history (last 4 turns, truncated to 300 chars each) is included for follow-up question context.

### `GET /books/{book_id}/chat`
Full chat history for this book (chronological), used to render the conversation on page load.
- **200**: `[{ "role": "user"|"assistant", "content": "...", "created_at": "..." }, ...]`
- **404**: book not found / not owned by caller

---

## Health

### `GET /health`
- **200**: `{ "status": "ok" }` — used for uptime checks / container health probes.

---

## Status code summary

| Code | Meaning here |
|---|---|
| 200 | Successful read |
| 201 | Signup created |
| 202 | Upload accepted, processing async |
| 204 | Delete succeeded |
| 401 | Missing/invalid/expired JWT |
| 404 | Resource doesn't exist or isn't yours |
| 409 | Conflict (duplicate email / book not ready for Q&A) |
| 413 | Upload too large |
| 415 | Unsupported file type |
| 422 | Request body failed validation |
| 500 | Unhandled server error (logged with a request id) |
