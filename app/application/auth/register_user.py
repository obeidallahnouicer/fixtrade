"""Register user use case."""

from app.application.auth.dtos import AuthSessionResult, RegisterUserCommand, UserResult
from app.domain.auth.entities import User
from app.domain.auth.errors import UserAlreadyExistsError
from app.domain.auth.ports import PasswordHasher, TokenService, UserRepository


class RegisterUserUseCase:
    """Create a new user account and issue an access token."""

    def __init__(
        self,
        user_repo: UserRepository,
        password_hasher: PasswordHasher,
        token_service: TokenService,
    ) -> None:
        self._user_repo = user_repo
        self._password_hasher = password_hasher
        self._token_service = token_service

    def execute(self, command: RegisterUserCommand) -> AuthSessionResult:
        existing_user = self._user_repo.get_by_email(command.email.lower())
        if existing_user is not None:
            raise UserAlreadyExistsError(command.email)

        user = User(
            id="",
            email=command.email.lower(),
            hashed_password=self._password_hasher.hash(command.password),
            full_name=command.full_name,
            role="user",
            is_active=True,
            created_at=None,  # populated by persistence layer
        )
        saved_user = self._user_repo.create(user)
        access_token, expires_in = self._token_service.create_access_token(saved_user)

        return AuthSessionResult(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResult(
                id=saved_user.id,
                email=saved_user.email,
                full_name=saved_user.full_name,
                role=saved_user.role,
                is_active=saved_user.is_active,
                created_at=saved_user.created_at,
            ),
        )