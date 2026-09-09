from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import (
    create_access_token, get_or_create_anonymous_user, hash_password, verify_password
)
from app.models.models import User
from app.models.schemas import TokenResponse, SignupRequest, LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/anonymous", response_model=TokenResponse)
def anonymous_session(db: Session = Depends(get_db)):
    """Called once by the frontend on first load in a browser tab/window.
    The returned token is stored client-side and scopes all subsequent
    requests to this 'session user' -> satisfies requirement #5."""
    user = get_or_create_anonymous_user(db)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = User(email=payload.email, hashed_password=hash_password(payload.password), is_anonymous=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id))
