"""Auth domain errors."""


class AuthError(Exception):
    """Base auth error."""


class UserAlreadyExistsError(AuthError):
    """Raised when trying to register an email that already exists."""


class InvalidCredentialsError(AuthError):
    """Raised when login credentials are invalid."""


class UserNotFoundError(AuthError):
    """Raised when a user cannot be loaded from storage."""