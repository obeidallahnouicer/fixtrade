"""Auth API schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(
        ..., min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: str = Field(
        ..., min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse