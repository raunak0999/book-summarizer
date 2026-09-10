import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.core.logging import log
from app.core.security import get_current_user
from app.models.models import Book, ChatMessage, User
from app.models.schemas import QuestionRequest, AnswerResponse, ChatMessageOut
from app.services.rag_service import retrieve_similar_chunks
from app.agents.graph import make_qa_graph

router = APIRouter(prefix="/books/{book_id}/chat", tags=["chat"])


def _load_book_or_404(db: Session, book_id: str, user: User) -> Book:
    book = db.get(Book, book_id)
    if not book or book.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


@router.post("", response_model=AnswerResponse)
async def ask_question(
    book_id: str,
    payload: QuestionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    book = _load_book_or_404(db, book_id, user)
    if book.status != "ready":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                             detail=f"Book is not ready yet (status={book.status})")

    history_rows = db.execute(
        select(ChatMessage).where(ChatMessage.book_id == book_id).order_by(ChatMessage.created_at)
    ).scalars().all()
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    def retriever(bid, query, k):
        return retrieve_similar_chunks(db, bid, query, k)

    graph = make_qa_graph(retriever)
    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {
                "book_id": book_id, "question": payload.question, "history": history,
                "retrieved": [], "relevant": False, "answer": "",
            }
        )
        db.add(ChatMessage(book_id=book_id, role="user", content=payload.question))
        db.add(ChatMessage(book_id=book_id, role="assistant", content=result["answer"]))
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        log.exception("chat_endpoint_failed", book_id=book_id, error=str(e))
        return AnswerResponse(
            answer="An error occurred while generating an answer for your question. Please try asking again.",
            sources=[],
        )

    sources = [r["ordinal"] for r in result["retrieved"]]
    return AnswerResponse(answer=result["answer"], sources=sources)


@router.get("", response_model=list[ChatMessageOut])
def get_history(book_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _load_book_or_404(db, book_id, user)
    rows = db.execute(
        select(ChatMessage).where(ChatMessage.book_id == book_id).order_by(ChatMessage.created_at)
    ).scalars().all()
    return rows
