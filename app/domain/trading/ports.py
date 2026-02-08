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
    ArticleSentiment,
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
