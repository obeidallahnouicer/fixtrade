"""Auth dependency wiring."""

from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.application.auth.login_user import LoginUserUseCase
from app.application.auth.register_user import RegisterUserUseCase
from app.core.db import get_db
from app.domain.auth.entities import User
from app.domain.auth.errors import InvalidCredentialsError, UserNotFoundError
from app.infrastructure.auth.repository import SQLAlchemyUserRepository
from app.infrastructure.auth.security import BcryptPasswordHasher, JwtTokenService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_user_repository(db: Session = Depends(get_db)) -> SQLAlchemyUserRepository:
    return SQLAlchemyUserRepository(session=db)


def get_password_hasher() -> BcryptPasswordHasher:
    return BcryptPasswordHasher()


def get_token_service() -> JwtTokenService:
    return JwtTokenService()


def get_register_user_use_case(
    user_repo: SQLAlchemyUserRepository = Depends(get_user_repository),
    password_hasher: BcryptPasswordHasher = Depends(get_password_hasher),
    token_service: JwtTokenService = Depends(get_token_service),
) -> RegisterUserUseCase:
    return RegisterUserUseCase(
        user_repo=user_repo,
        password_hasher=password_hasher,
        token_service=token_service,
    )


def get_login_user_use_case(
    user_repo: SQLAlchemyUserRepository = Depends(get_user_repository),
    password_hasher: BcryptPasswordHasher = Depends(get_password_hasher),
    token_service: JwtTokenService = Depends(get_token_service),
) -> LoginUserUseCase:
    return LoginUserUseCase(
        user_repo=user_repo,
        password_hasher=password_hasher,
        token_service=token_service,
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    token_service: JwtTokenService = Depends(get_token_service),
    user_repo: SQLAlchemyUserRepository = Depends(get_user_repository),
) -> User:
    try:
        payload = token_service.decode_access_token(token)
    except Exception as exc:  # pragma: no cover - converted to HTTP error below
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user