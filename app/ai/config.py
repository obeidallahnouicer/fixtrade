"""
AI module configuration.

Manages settings for:
- Groq API integration
- Risk profile parameters
- Portfolio simulation settings
- Recommendation thresholds
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AISettings(BaseSettings):
    """AI module settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # --- Groq API Configuration ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.7
    
    # --- Portfolio Defaults ---
    default_initial_capital: float = 10000.0  # TND
    default_risk_profile: str = "moderate"
    
    # --- Risk Profile Thresholds ---
    conservative_max_position_size: float = 0.10  # Max 10% per position
    moderate_max_position_size: float = 0.15      # Max 15% per position
    aggressive_max_position_size: float = 0.25    # Max 25% per position
    
    conservative_max_equity_allocation: float = 0.40  # Max 40% in stocks
    moderate_max_equity_allocation: float = 0.70      # Max 70% in stocks
    aggressive_max_equity_allocation: float = 0.90    # Max 90% in stocks
    
    # --- Recommendation Thresholds ---
    min_confidence_score: float = 0.65
    min_sentiment_score: float = 0.20
    anomaly_severity_threshold: float = 0.75
    
    # --- Trading Rules ---
    min_holding_days: int = 1
    max_daily_trades: int = 5
    stop_loss_conservative: float = 0.05   # 5% stop-loss
    stop_loss_moderate: float = 0.08       # 8% stop-loss
    stop_loss_aggressive: float = 0.12     # 12% stop-loss
    
    # --- Performance Metrics ---
    risk_free_rate: float = 0.05  # 5% annual risk-free rate (TND bonds)
    trading_days_per_year: int = 250


ai_settings = AISettings()
