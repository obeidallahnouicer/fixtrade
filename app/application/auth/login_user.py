"""Login user use case."""

from app.application.auth.dtos import AuthSessionResult, LoginUserCommand, UserResult
from app.domain.auth.errors import InvalidCredentialsError
from app.domain.auth.ports import PasswordHasher, TokenService, UserRepository


class LoginUserUseCase:
    """Authenticate a user and issue an access token."""

    def __init__(
        self,
        user_repo: UserRepository,
        password_hasher: PasswordHasher,
        token_service: TokenService,
    ) -> None:
        self._user_repo = user_repo
        self._password_hasher = password_hasher
        self._token_service = token_service

    def execute(self, command: LoginUserCommand) -> AuthSessionResult:
        user = self._user_repo.get_by_email(command.email.lower())
        if user is None or not user.is_active:
            raise InvalidCredentialsError()

        if not self._password_hasher.verify(command.password, user.hashed_password):
            raise InvalidCredentialsError()

        access_token, expires_in = self._token_service.create_access_token(user)

        return AuthSessionResult(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResult(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
            ),
        )