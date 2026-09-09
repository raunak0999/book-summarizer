from pydantic import BaseModel, EmailStr
from datetime import datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class BookOut(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int | None
    summary_100w: str | None
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    question: str


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnswerResponse(BaseModel):
    answer: str
    sources: list[int] = []
