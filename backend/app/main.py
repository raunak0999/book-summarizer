from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from sqlalchemy import text
from openai import RateLimitError, APIError

from app.core.logging import configure_logging, RequestContextMiddleware, log
from app.core.db import Base, engine
from app.api import auth, books, chat

configure_logging()

app = FastAPI(
    title="Book Summarizer & Query Agent",
    description="RAG-based agentic book summarizer + Q&A API",
    version="1.0.0",
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Enable pgvector extension + create tables if they don't exist.
    try:
        with engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                engine.dialect.has_pgvector = True
            except Exception as e:
                conn.rollback()
                engine.dialect.has_pgvector = False
                log.warning("pgvector_extension_failed_using_fallback", error=str(e))
                conn.execute(text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') THEN
                            CREATE DOMAIN vector AS float8[];
                        END IF;
                    END $$;
                    CREATE OR REPLACE FUNCTION vector_cosine_distance(a vector, b vector) RETURNS float8 AS $fn$
                    DECLARE
                        dot_product float8 := 0;
                        norm_a float8 := 0;
                        norm_b float8 := 0;
                        i int;
                    BEGIN
                        FOR i IN 1..cardinality(a) LOOP
                            dot_product := dot_product + (a[i] * b[i]);
                            norm_a := norm_a + (a[i] * a[i]);
                            norm_b := norm_b + (b[i] * b[i]);
                        END LOOP;
                        IF norm_a = 0 OR norm_b = 0 THEN
                            RETURN 1.0;
                        END IF;
                        RETURN 1.0 - (dot_product / (sqrt(norm_a) * sqrt(norm_b)));
                    END;
                    $fn$ LANGUAGE plpgsql IMMUTABLE STRICT;
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<=>' AND oprleft = 'vector'::regtype) THEN
                            CREATE OPERATOR <=> (
                                LEFTARG = vector,
                                RIGHTARG = vector,
                                FUNCTION = vector_cosine_distance
                            );
                        END IF;
                    END $$;
                """))
                conn.commit()
        Base.metadata.create_all(bind=engine)
    except Exception as db_err:
        log.warning("db_startup_initialization_warning", error=str(db_err))
    try:
        from app.services.llm_client import get_llm_client, _get_st_model
        llm = get_llm_client()
        if getattr(llm, "embedding_provider", None) == "local":
            _get_st_model()
            log.info("local_embedding_model_warmed_up")
        else:
            log.info("remote_embedding_provider_no_warmup_needed", provider=getattr(llm, 'embedding_provider', 'unknown'))
        log.info("chat_model_configured", provider=getattr(llm, 'chat_provider', 'unknown'))
    except Exception as e:
        log.warning("embedding_warmup_failed", error=str(e))

    # Auto-recover books stuck in "processing" after a redeploy
    try:
        from datetime import datetime, timezone, timedelta
        from app.core.db import SessionLocal
        from app.models.models import Book
        with SessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
            stuck = db.query(Book).filter(
                Book.status == "processing",
                Book.created_at < cutoff,
            ).all()
            for book in stuck:
                book.status = "failed"
                book.error_message = "Processing was interrupted by a server restart. Please delete this book and re-upload."
                log.warning("auto_recovered_stuck_book", book_id=book.id, filename=book.filename)
            if stuck:
                db.commit()
                log.info("stuck_books_recovered", count=len(stuck))
    except Exception as e:
        log.warning("stuck_book_recovery_failed", error=str(e))

    log.info("startup_complete")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.warning("validation_error", errors=exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                         content={"detail": exc.errors()})


@app.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
    log.warning("rate_limit_error", error=str(exc))
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                         content={"detail": f"Rate limit reached on LLM provider: {str(exc)}"})


@app.exception_handler(APIError)
async def openai_api_exception_handler(request: Request, exc: APIError):
    log.warning("openai_api_error", error=str(exc))
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY,
                         content={"detail": f"LLM provider error: {str(exc)}"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("unhandled_exception", error=str(exc))
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         content={"detail": "Internal server error"})


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


app.include_router(auth.router)
app.include_router(books.router)
app.include_router(chat.router)
