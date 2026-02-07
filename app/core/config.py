"""
Application configuration.

Loads settings from environment variables and .env file.
All configuration is centralized here — no scattered magic strings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment.

    Attributes:
        project_name: Display name for the API.
        version: Current API version string.
        debug: Enable debug mode. Must be False in production.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        rate_limit_default: Default rate limit for all endpoints.
        rate_limit_heavy: Rate limit for compute-heavy endpoints.
        max_request_size_bytes: Maximum allowed request body size.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    project_name: str = "FixTrade"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    rate_limit_default: str = "60/minute"
    rate_limit_heavy: str = "10/minute"
    max_request_size_bytes: int = 1_048_576  # 1 MB


settings = Settings()
