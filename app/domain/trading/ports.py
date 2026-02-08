"""
Port interfaces (ABCs) for the trading bounded context.

Ports define the contracts that the domain requires from the outside world.
Infrastructure adapters implement these interfaces.
The domain layer never depends on concrete implementations.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from app.domain.trading.entities import (
    AnomalyAlert,
    ArticleSentiment,
    IntradayTick,
    LiquidityForecast,
    Portfolio,
    PricePrediction,
    ScrapedArticle,
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


class IntradayTickRepository(ABC):
    """Port for retrieving and persisting intraday tick data.

    Supports 1-minute bars and raw tick-by-tick records for
    high-frequency anomaly detection.
    """

    @abstractmethod
    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tick_type: str = "1min",
    ) -> list[IntradayTick]:
        """Return intraday ticks for a symbol within the datetime range.

        Args:
            symbol: BVMT stock ticker.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            tick_type: Filter by tick type ("1min" or "tick").

        Returns:
            List of IntradayTick ordered by timestamp ascending.
        """
        raise NotImplementedError

    @abstractmethod
    def save_batch(self, ticks: list[IntradayTick]) -> int:
        """Persist a batch of intraday ticks.

        Args:
            ticks: List of IntradayTick entities.

        Returns:
            Number of rows inserted (duplicates ignored).
        """
        raise NotImplementedError

    @abstractmethod
    def get_symbols_with_data(self, since: datetime) -> list[str]:
        """Return symbols that have intraday data since a given datetime."""
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
        raise NotImplementedError

    @abstractmethod
    def analyze_text(self, text: str) -> int:
        """Run NLP inference on raw text and return a sentiment score.

        Returns:
            1 for positive, -1 for negative, 0 for neutral.
        """
        raise NotImplementedError


class AnomalyDetectionPort(ABC):
    """Port for detecting market anomalies."""

    @abstractmethod
    def detect(self, symbol: str) -> list[AnomalyAlert]:
        """Return list of detected anomalies for a symbol."""
        # TODO: implement in infrastructure adapter
        raise NotImplementedError


class AnomalyAlertRepository(ABC):
    """Port for persisting and retrieving anomaly alerts."""

    @abstractmethod
    def save(self, alert: AnomalyAlert) -> None:
        """Persist a single anomaly alert."""
        raise NotImplementedError

    @abstractmethod
    def save_batch(self, alerts: list[AnomalyAlert]) -> None:
        """Persist multiple anomaly alerts."""
        raise NotImplementedError

    @abstractmethod
    def get_recent(
        self, symbol: Optional[str] = None, limit: int = 10, since: Optional[datetime] = None
    ) -> list[AnomalyAlert]:
        """Return recent anomaly alerts.

        Args:
            symbol: Optional filter by stock symbol.
            limit: Maximum number of alerts to return.
            since: Optional filter for alerts detected after this datetime.

        Returns:
            List of anomaly alerts ordered by detected_at descending.
        """
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


class ScrapedArticleRepository(ABC):
    """Port for reading scraped articles from storage."""

    @abstractmethod
    def get_unanalyzed_articles(self, limit: int = 100) -> list[ScrapedArticle]:
        """Return articles that have not yet been sentiment-analyzed."""
        raise NotImplementedError

    @abstractmethod
    def get_article_text(self, article: ScrapedArticle) -> str:
        """Return the best available text for NLP analysis.

        Prefers content, falls back to summary, then title.
        """
        raise NotImplementedError


class ArticleSentimentRepository(ABC):
    """Port for persisting per-article sentiment analysis results."""

    @abstractmethod
    def save(self, result: ArticleSentiment) -> None:
        """Persist a single article sentiment result."""
        raise NotImplementedError

    @abstractmethod
    def save_batch(self, results: list[ArticleSentiment]) -> None:
        """Persist multiple article sentiment results."""
        raise NotImplementedError
