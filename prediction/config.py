"""
Prediction module configuration.

Reads infrastructure settings (DB, Redis, paths) from the app's
central Settings object (app.core.config), which loads from .env.

ML-specific constants (model hyperparameters, feature windows,
liquidity tiers) are defined here — they don't belong in .env
because they are tuned by experimentation, not by deployment.
"""

from dataclasses import dataclass, field
from pathlib import Path


def _load_app_settings():
    """Lazy-load the app settings to avoid circular imports."""
    try:
        from app.core.config import settings
        return settings
    except Exception:
        return None


@dataclass(frozen=True)
class PathsConfig:
    """File system paths for the Medallion data architecture."""

    base_dir: Path = field(default_factory=lambda: Path(
        _s.fixtrade_data_dir if (_s := _load_app_settings()) else "data"
    ))

    @property
    def bronze(self) -> Path:
        return self.base_dir / "bronze"

    @property
    def silver(self) -> Path:
        return self.base_dir / "silver"

    @property
    def gold(self) -> Path:
        return self.base_dir / "gold"

    @property
    def raw(self) -> Path:
        return self.base_dir / "raw"

    @property
    def models_dir(self) -> Path:
        s = _load_app_settings()
        if s and hasattr(s, "model_dir"):
            return Path(s.model_dir)
        return self.base_dir.parent / "models"


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection settings — read from .env via app settings."""

    host: str = field(default_factory=lambda: (
        _s.postgres_host if (_s := _load_app_settings()) else "localhost"
    ))
    port: int = field(default_factory=lambda: (
        _s.postgres_port if (_s := _load_app_settings()) else 5432
    ))
    database: str = field(default_factory=lambda: (
        _s.postgres_db if (_s := _load_app_settings()) else "fixtrade"
    ))
    user: str = field(default_factory=lambda: (
        _s.postgres_user if (_s := _load_app_settings()) else "postgres"
    ))
    password: str = field(default_factory=lambda: (
        _s.postgres_password if (_s := _load_app_settings()) else ""
    ))

    @property
    def url(self) -> str:
        s = _load_app_settings()
        if s and hasattr(s, "database_url"):
            return s.database_url
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection settings — read from .env via app settings."""

    host: str = field(default_factory=lambda: (
        _s.redis_host if (_s := _load_app_settings()) else "localhost"
    ))
    port: int = field(default_factory=lambda: (
        _s.redis_port if (_s := _load_app_settings()) else 6379
    ))
    db: int = field(default_factory=lambda: (
        _s.redis_db if (_s := _load_app_settings()) else 0
    ))
    password: str | None = field(default_factory=lambda: (
        (_s.redis_password or None) if (_s := _load_app_settings()) else None
    ))
    feature_ttl_days: int = 90
    prediction_ttl_seconds: int = field(default_factory=lambda: (
        _s.prediction_cache_ttl if (_s := _load_app_settings()) else 3600
    ))
    post_market_ttl_seconds: int = 43200  # 12 hours

    @property
    def url(self) -> str:
        s = _load_app_settings()
        if s and hasattr(s, "redis_url"):
            return s.redis_url
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class ModelConfig:
    """ML model hyperparameters and training settings."""

    # LSTM
    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_patience: int = 10

    # XGBoost
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping_rounds: int = 20

    # Prophet
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True

    # Ensemble default weights (Phase 1 MVP)
    ensemble_lstm_weight: float = 0.45
    ensemble_xgb_weight: float = 0.35
    ensemble_prophet_weight: float = 0.20

    # Training
    train_test_split_year: int = 2025
    validation_year: int = 2024
    min_training_samples: int = 252  # ~1 year of trading days


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters."""

    # Technical indicator windows
    sma_windows: tuple[int, ...] = (5, 10, 20, 50, 200)
    ema_windows: tuple[int, ...] = (12, 26)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_window: int = 14
    stochastic_window: int = 14

    # Lag features
    lag_days: tuple[int, ...] = (1, 2, 3, 5, 10, 20)

    # Volume features
    volume_sma_windows: tuple[int, ...] = (5, 10, 20)
    vwap_window: int = 20

    # Target horizons (days ahead)
    prediction_horizons: tuple[int, ...] = (1, 2, 3, 5)


@dataclass(frozen=True)
class LiquidityTierConfig:
    """Liquidity tier thresholds for model selection."""

    tier1_min_volume: int = 10_000  # High liquidity
    tier2_min_volume: int = 1_000   # Medium liquidity
    # Below tier2 → Tier 3 (low liquidity)


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow experiment tracking settings."""

    tracking_uri: str = field(default_factory=lambda: (
        _s.mlflow_tracking_uri if (_s := _load_app_settings()) else "mlruns"
    ))
    experiment_name: str = field(default_factory=lambda: (
        _s.mlflow_experiment_name if (_s := _load_app_settings()) else "fixtrade-prediction"
    ))


@dataclass(frozen=True)
class PredictionConfig:
    """Top-level configuration aggregating all sub-configs."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    liquidity: LiquidityTierConfig = field(default_factory=LiquidityTierConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # BVMT tickers to track
    tracked_tickers: tuple[str, ...] = field(default_factory=lambda: (
        "BIAT", "BH", "BNA", "STB", "BT", "UBCI", "UIB", "ATB",
        "SFBT", "DELICE", "POULINA", "SAH", "TPR", "SOTET",
        "TELNET", "ARTES", "OTH", "SITS", "SOPAT", "SIMPAR",
        "STAR", "ASTREE", "WIFACK", "TJARI", "AETECH",
        "CELLCOM", "CIMENTS", "CARTHAGE", "MONOPRIX", "TUNAIR",
    ))


# Singleton instance
config = PredictionConfig()
