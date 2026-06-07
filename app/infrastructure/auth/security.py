"""Password hashing and JWT token services for auth."""

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.domain.auth.entities import User
from app.domain.auth.ports import PasswordHasher, TokenService


class BcryptPasswordHasher(PasswordHasher):
    """Hash passwords with bcrypt."""

    def __init__(self) -> None:
        self._context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash(self, password: str) -> str:
        return self._context.hash(password)

    def verify(self, plain_password: str, hashed_password: str) -> bool:
        return self._context.verify(plain_password, hashed_password)


class JwtTokenService(TokenService):
    """Create and validate JWT access tokens."""

    def __init__(self) -> None:
        self._secret_key = settings.auth_secret_key
        self._algorithm = settings.auth_algorithm
        self._expires_minutes = settings.access_token_expire_minutes

    def create_access_token(self, user: User) -> tuple[str, int]:
        expires_delta = timedelta(minutes=self._expires_minutes)
        expire_at = datetime.now(timezone.utc) + expires_delta
        payload = {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "type": "access",
            "exp": expire_at,
            "iat": datetime.now(timezone.utc),
            "iss": "fixtrade",
        }
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        return token, self._expires_minutes

    def decode_access_token(self, token: str) -> dict:
        return jwt.decode(
            token,
            self._secret_key,
            algorithms=[self._algorithm],
            options={"require": ["exp", "sub"]},
            issuer="fixtrade",
        )