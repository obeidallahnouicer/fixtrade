"""Password hashing and JWT token services for auth."""

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.domain.auth.entities import User
from app.domain.auth.ports import PasswordHasher, TokenService


class BcryptPasswordHasher(PasswordHasher):
    """Hash passwords using bcrypt_sha256 to avoid bcrypt's 72-byte limit.

    `bcrypt` has a 72-byte input limit which raises a ValueError for longer
    secrets. `bcrypt_sha256` pre-hashes the password with SHA-256 before
    applying bcrypt which safely supports arbitrary-length passwords.
    """

    def __init__(self) -> None:
        # Prefer a pure-Python scheme to avoid C-extension mismatch issues
        # and bcrypt's 72-byte limit. `pbkdf2_sha256` supports arbitrary
        # password lengths and is widely available. Keep bcrypt variants
        # as fallbacks for existing hashes.
        self._context = CryptContext(
            schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"],
            deprecated="auto",
        )

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