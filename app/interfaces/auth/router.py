"""FastAPI router for auth endpoints."""

from fastapi import APIRouter, Depends, Request

from app.application.auth.dtos import LoginUserCommand, RegisterUserCommand
from app.domain.auth.errors import InvalidCredentialsError, UserAlreadyExistsError
from app.interfaces.auth.dependencies import (
    get_current_user,
    get_login_user_use_case,
    get_register_user_use_case,
)
from app.interfaces.auth.schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse
from app.shared.security.rate_limiting import limiter

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.post(
    "/register",
    response_model=AuthResponse,
    summary="Register a new user",
)
@limiter.limit("5/minute")
def register(
    request: Request,
    payload: RegisterRequest,
    use_case=Depends(get_register_user_use_case),
) -> AuthResponse:
    result = use_case.execute(
        RegisterUserCommand(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
        )
    )
    return AuthResponse(
        access_token=result.access_token,
        token_type=result.token_type,
        expires_in=result.expires_in,
        user=_to_user_response(result.user),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login an existing user",
)
@limiter.limit("10/minute")
def login(
    request: Request,
    payload: LoginRequest,
    use_case=Depends(get_login_user_use_case),
) -> AuthResponse:
    result = use_case.execute(
        LoginUserCommand(email=payload.email, password=payload.password)
    )
    return AuthResponse(
        access_token=result.access_token,
        token_type=result.token_type,
        expires_in=result.expires_in,
        user=_to_user_response(result.user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user",
)
def me(current_user=Depends(get_current_user)) -> UserResponse:
    return _to_user_response(current_user)