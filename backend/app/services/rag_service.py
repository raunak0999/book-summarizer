import math
import time

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.models import Chunk
from app.services.llm_client import get_llm_client
from app.services.document_processing import count_tokens
from app.core.logging import log

# Delay between embedding API calls to stay under free-tier RPM limits.
# Gemini embedding quota is typically 10-15 RPM; 4s keeps us safely under.
INTER_BATCH_DELAY_S = 4.0


def store_chunks(db: Session, book_id: str, chunks: list[str]):
    llm = get_llm_client()
    # Batch embeddings to limit API calls / cost.
    batch_size = 32
    total_batches = math.ceil(len(chunks) / batch_size)
    for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[start:start + batch_size]
        log.info(
            "embedding_batch_start",
            book_id=book_id,
            batch=f"{batch_idx + 1}/{total_batches}",
            chunk_range=f"{start}-{start + len(batch) - 1}",
        )
        embeddings = llm.embed(batch)
        for i, (text, emb) in enumerate(zip(batch, embeddings)):
            db.add(Chunk(
                book_id=book_id,
                ordinal=start + i,
                content=text,
                token_count=count_tokens(text),
                embedding=emb,
            ))
        db.commit()
        log.info(
            "embedding_batch_complete",
            book_id=book_id,
            batch=f"{batch_idx + 1}/{total_batches}",
        )
        # Pace requests to avoid hitting RPM quota — skip delay after last batch
        if batch_idx < total_batches - 1:
            time.sleep(INTER_BATCH_DELAY_S)
    log.info("chunks_stored", book_id=book_id, n_chunks=len(chunks))


def retrieve_similar_chunks(db: Session, book_id: str, query: str, k: int) -> list[dict]:
    llm = get_llm_client()
    query_embedding = llm.embed([query])[0]

    # pgvector cosine distance operator `<=>`; convert to a 0..1-ish
    # similarity score for the relevance-grading heuristic.
    stmt = (
        select(Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .where(Chunk.book_id == book_id)
        .order_by("distance")
        .limit(k)
    )
    rows = db.execute(stmt).all()
    return [
        {"ordinal": chunk.ordinal, "content": chunk.content, "score": 1 - distance}
        for chunk, distance in rows
    ]
