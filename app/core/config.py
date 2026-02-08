"""
Application configuration.

Single source of truth: reads everything from .env file.
No hardcoded values — all defaults live in .env only.
Both the FastAPI app and the prediction module import from here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


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

    # --- MLflow ---
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "fixtrade-prediction"

    # --- AI Agent (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.7

    # --- AI / Groq ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Rate Limiting ---
    rate_limit_default: str
    rate_limit_heavy: str

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
