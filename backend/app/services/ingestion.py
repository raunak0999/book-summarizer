from app.core.db import SessionLocal
from app.models.models import Book
from app.services.document_processing import extract_text, chunk_text
from app.services.rag_service import store_chunks
from app.agents.graph import run_summarization
from app.core.config import get_settings
from app.core.logging import log

settings = get_settings()


def process_book(book_id: str, file_path: str, filename: str):
    """Runs synchronously in a background task. Creates its own DB session."""
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
                book.status = "failed"
                book.error_message = f"{str(e)}\n\nTraceback:\n{tb_str}"
                db.commit()


