from app.core.db import SessionLocal
from app.models.models import Book, Chunk
from app.services.document_processing import extract_text, chunk_text
from app.services.rag_service import store_chunks
from app.agents.graph import run_summarization
from app.core.config import get_settings
from app.core.logging import log

settings = get_settings()
from app.services.llm_client import _is_rate_limit_error



def _count_existing_chunks(db, book_id: str) -> int:
    """Count how many chunks are already committed for this book."""
    from sqlalchemy import func, select
    result = db.execute(select(func.count()).where(Chunk.book_id == book_id)).scalar()
    return result or 0


def process_book(book_id: str, file_path: str, filename: str):
    """Runs synchronously in a background task. Creates its own DB session."""
    import structlog
    structlog.contextvars.bind_contextvars(path="background_ingestion", book_id=book_id)
    with SessionLocal() as db:
        book = db.get(Book, book_id)
        if not book:
            log.error("book_not_found_for_processing", book_id=book_id)
            return
        try:
            full_text, page_count = extract_text(file_path, filename)
            if not full_text.strip():
                raise ValueError("No extractable text found in file (possibly a scanned/image PDF).")

            book.page_count = page_count
            db.commit()

            chunks = chunk_text(full_text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
            store_chunks(db, book_id, chunks)

            summary = run_summarization(full_text)
            book.summary_100w = summary
            book.status = "ready"
            db.commit()
            log.info("book_processed", book_id=book_id, pages=page_count, chunks=len(chunks))
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            db.rollback()
            log.error("book_processing_failed", book_id=book_id, error=str(e), traceback=tb_str)
            book = db.get(Book, book_id)
            if book:
                if _is_rate_limit_error(e):
                    saved = _count_existing_chunks(db, book_id)
                    book.status = "rate_limited"
                    book.error_message = (
                        f"Processing paused — API rate limit reached after embedding "
                        f"{saved} chunks. Previously embedded chunks are saved. "
                        f"Please delete and re-upload to retry later."
                    )
                else:
                    book.status = "failed"
                    # Clean one-line message for the user; full traceback stays in server logs
                    book.error_message = str(e)
                db.commit()
        finally:
            import os
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
