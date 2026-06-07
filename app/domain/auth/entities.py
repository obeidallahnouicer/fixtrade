"""Auth domain entities."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class User:
    """Authenticated user entity."""

    id: str
    email: str
    hashed_password: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime | None