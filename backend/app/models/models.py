"""
Storage strategy (see /docs/architecture.md for the full rationale):

- Book: one row per uploaded book, holds a *book-level* summary embedding
  (a "coarse" vector representing the whole document) generated from the
  100-word summary itself, used to power "find similar books" later and
  fast dedup checks.
- Chunk: the book is split into overlapping ~800-token chunks (chunking
  strategy chosen over "one vector per book" because 500+ page books
  blow past any embedding model's useful context and hurt retrieval
  precision). Each chunk gets its own embedding row in pgvector.
- ChatMessage: full Q&A history per book per session, so "history" is
  just a query away and the RAG agent can also use recent turns as
  context for follow-up questions.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType
from pgvector.sqlalchemy import Vector as PGVector

from app.core.db import Base
from app.core.config import get_settings

class Vector(UserDefinedType):
    cache_ok = True
    comparator_factory = PGVector.comparator_factory

    def get_col_spec(self, **kw):
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            has_pgvector = getattr(dialect, "has_pgvector", False)
            if isinstance(value, (list, tuple)):
                if has_pgvector:
                    return f"[{','.join(str(float(x)) for x in value)}]"
                else:
                    return f"{{{','.join(str(float(x)) for x in value)}}}"
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                cleaned = value.strip("{}[]")
                return [float(x) for x in cleaned.split(",") if x.strip()]
            return value
        return process

EMBED_DIM = get_settings().embedding_dim


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=True)
    # Anonymous/browser-session users (requirement #5) don't need an email/password.
    is_anonymous: Mapped[bool] = mapped_column(default=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=True)
    session_token: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    books: Mapped[list["Book"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Book(Base):
    __tablename__ = "books"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="processing")  # processing|ready|failed
    page_count: Mapped[int] = mapped_column(Integer, nullable=True)
    summary_100w: Mapped[str] = mapped_column(Text, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owner: Mapped["User"] = relationship(back_populates="books")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="book", cascade="all, delete-orphan")


class Chunk(Base):
    """One retrieval unit. Sequential ordinal preserved so we can expand
    context to neighboring chunks when a match is found (small-to-big
    retrieval)."""
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding = mapped_column(Vector)

    book: Mapped["Book"] = relationship(back_populates="chunks")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    book: Mapped["Book"] = relationship(back_populates="messages")
