import os
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.core.security import get_current_user
from app.core.config import get_settings
from app.models.models import Book, User
from app.models.schemas import BookOut
from app.services.ingestion import process_book

router = APIRouter(prefix="/books", tags=["books"])
settings = get_settings()

ALLOWED_EXT = {".pdf", ".docx", ".txt"}


@router.post("", response_model=BookOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                             detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXT}")

    os.makedirs(settings.upload_dir, exist_ok=True)
    saved_name = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(settings.upload_dir, saved_name)

    size = 0
    with open(saved_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                os.remove(saved_path)
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                     detail=f"File exceeds {settings.max_upload_mb}MB limit")
            out.write(chunk)

    book = Book(owner_id=user.id, filename=file.filename, status="processing")
    db.add(book)
    db.commit()
    db.refresh(book)

    # Runs after the response is sent; status is polled via GET /books/{id}.
    background_tasks.add_task(process_book, book.id, saved_path, file.filename)

    return book


@router.get("", response_model=list[BookOut])
def list_books(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """The 'history of other books extracted' requirement — scoped strictly
    to the current user via owner_id, so no cross-user data leakage."""
    stmt = select(Book).where(Book.owner_id == user.id).order_by(Book.created_at.desc())
    return db.execute(stmt).scalars().all()


@router.get("/{book_id}", response_model=BookOut)
def get_book(book_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    book = db.get(Book, book_id)
    if not book or book.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    book = db.get(Book, book_id)
    if not book or book.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    db.delete(book)
    db.commit()
    return None
