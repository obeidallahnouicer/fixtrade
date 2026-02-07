"""
Application configuration.

Single source of truth: reads everything from .env file.
No hardcoded values — all defaults live in .env only.
Both the FastAPI app and the prediction module import from here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded entirely from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Application ---
    project_name: str
    version: str
    debug: bool
    log_level: str

    # --- PostgreSQL ---
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    database_url: str

    # --- Redis ---
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: str
    redis_url: str

    # --- Data Paths ---
    fixtrade_data_dir: str

    # --- ML / Prediction ---
    prediction_cache_ttl: int
    model_dir: str

    # --- Rate Limiting ---
    rate_limit_default: str
    rate_limit_heavy: str


settings = Settings()
