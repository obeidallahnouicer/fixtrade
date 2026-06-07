"""Auth domain ports."""

from abc import ABC, abstractmethod

from app.domain.auth.entities import User


class UserRepository(ABC):
    """Repository contract for user persistence."""

    @abstractmethod
    def get_by_email(self, email: str) -> User | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, user_id: str) -> User | None:
        raise NotImplementedError

    @abstractmethod
    def create(self, user: User) -> User:
        raise NotImplementedError


class PasswordHasher(ABC):
    """Password hashing contract."""

    @abstractmethod
    def hash(self, password: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def verify(self, plain_password: str, hashed_password: str) -> bool:
        raise NotImplementedError


class TokenService(ABC):
    """JWT token contract."""

    @abstractmethod
    def create_access_token(self, user: User) -> tuple[str, int]:
        raise NotImplementedError

    @abstractmethod
    def decode_access_token(self, token: str) -> dict:
        raise NotImplementedError