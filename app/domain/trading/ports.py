"""
Port interfaces (ABCs) for the trading bounded context.

Ports define the contracts that the domain requires from the outside world.
Infrastructure adapters implement these interfaces.
The domain layer never depends on concrete implementations.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional
from uuid import UUID

from app.domain.trading.entities import (
    AnomalyAlert,
    LiquidityForecast,
    Portfolio,
    PricePrediction,
    SentimentScore,
    StockPrice,
    TradeRecommendation,
    VolumePrediction,
)


class StockPriceRepository(ABC):
    """Port for retrieving historical stock price data."""

    @abstractmethod
    def get_history(
        self, symbol: str, start: date, end: date
    ) -> list[StockPrice]:
        """Return OHLCV history for a symbol within the date range."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError


class PricePredictionPort(ABC):
    """Port for obtaining price predictions from an ML model."""

    @abstractmethod
    def predict(
        self, symbol: str, horizon_days: int
    ) -> list[PricePrediction]:
        """Return predicted closing prices for the next N trading days."""
        raise NotImplementedError

    @abstractmethod
    def predict_volume(
        self, symbol: str, horizon_days: int
    ) -> list[VolumePrediction]:
        """Return predicted daily transaction volumes for the next N trading days."""
        raise NotImplementedError

    @abstractmethod
    def predict_liquidity(
        self, symbol: str, horizon_days: int
    ) -> list[LiquidityForecast]:
        """Return liquidity tier probabilities for the next N trading days."""
        raise NotImplementedError


class SentimentAnalysisPort(ABC):
    """Port for obtaining sentiment scores from the NLP pipeline."""

    @abstractmethod
    def get_sentiment(
        self, symbol: str, target_date: Optional[date] = None
    ) -> SentimentScore:
        """Return aggregated sentiment score for a symbol on a given date."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError


class AnomalyDetectionPort(ABC):
    """Port for detecting market anomalies."""

    @abstractmethod
    def detect(self, symbol: str) -> list[AnomalyAlert]:
        """Return list of detected anomalies for a symbol."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError


class PortfolioRepository(ABC):
    """Port for persisting and retrieving portfolios."""

    @abstractmethod
    def get_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """Return a portfolio by its ID, or None if not found."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError

    @abstractmethod
    def save(self, portfolio: Portfolio) -> None:
        """Persist a portfolio."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError


class DecisionEnginePort(ABC):
    """Port for generating trade recommendations."""

    @abstractmethod
    def recommend(self, symbol: str, portfolio_id: UUID) -> TradeRecommendation:
        """Return a buy/sell/hold recommendation for a symbol."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError
