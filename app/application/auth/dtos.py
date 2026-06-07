"""Auth application DTOs."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class RegisterUserCommand:
    email: str
    password: str
    full_name: str | None = None


@dataclass(slots=True)
class LoginUserCommand:
    email: str
    password: str


@dataclass(slots=True)
class UserResult:
    id: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime


@dataclass(slots=True)
class AuthSessionResult:
    access_token: str
    token_type: str
    expires_in: int
    user: UserResult