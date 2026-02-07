"""
Application configuration.

Loads settings from environment variables and .env file.
All configuration is centralized here — no scattered magic strings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


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

    Database/Postgres settings are included here so local development can use
    the same configuration source as the rest of the application.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    project_name: str = "FixTrade"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    rate_limit_default: str = "60/minute"
    rate_limit_heavy: str = "10/minute"
    max_request_size_bytes: int = 1_048_576  # 1 MB

    # Postgres settings (used by the scraping pipeline and other components)
    scraping_postgres_dsn: Optional[str] = None
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fixtrade_scraping"

    def get_scraping_postgres_dsn(self) -> str:
        """Return the effective DSN for scraping Postgres.

        Priority:
        1. Explicit `SCRAPING_POSTGRES_DSN` (set as env var and loaded in scraping settings)
        2. Build DSN from postgres_* values (useful for Docker Compose or local setups)
        """
        if self.scraping_postgres_dsn:
            return self.scraping_postgres_dsn
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
