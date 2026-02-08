"""
Domain entities for the trading bounded context.

Entities represent core business objects with identity and lifecycle.
They contain no framework imports and no IO operations.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class Sentiment(Enum):
    """Sentiment classification for a news article or aggregated score."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Recommendation(Enum):
    """Trading recommendation action."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskProfile(Enum):
    """Investor risk profile."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class StockPrice:
    """A single day's OHLCV price record for a BVMT-listed stock."""

    symbol: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True)
class PricePrediction:
    """Predicted closing price for a future trading day."""

    symbol: str
    target_date: date
    predicted_close: Decimal
    confidence_lower: Decimal
    confidence_upper: Decimal


@dataclass(frozen=True)
class VolumePrediction:
    """Predicted daily transaction volume for a future trading day."""

    symbol: str
    target_date: date
    predicted_volume: int


@dataclass(frozen=True)
class LiquidityForecast:
    """Probability distribution over liquidity tiers for a future trading day.

    Tiers:
        low   — volume < 1 000
        medium — 1 000 ≤ volume < 10 000
        high  — volume ≥ 10 000
    """

    symbol: str
    target_date: date
    prob_low: Decimal
    prob_medium: Decimal
    prob_high: Decimal

    @property
    def predicted_tier(self) -> str:
        """Return the tier with the highest probability."""
        probs = [
            (self.prob_low, "low"),
            (self.prob_medium, "medium"),
            (self.prob_high, "high"),
        ]
        return max(probs, key=lambda t: t[0])[1]


@dataclass(frozen=True)
class SentimentScore:
    """Aggregated daily sentiment score for a given stock."""

    symbol: str
    date: date
    score: Decimal
    sentiment: Sentiment
    article_count: int


@dataclass(frozen=True)
class ScrapedArticle:
    """A single scraped news article from the database."""

    id: int
    url: str
    title: Optional[str]
    summary: Optional[str]
    content: Optional[str]
    published_at: Optional[datetime]


@dataclass(frozen=True)
class ArticleSentiment:
    """Sentiment analysis result for a single scraped article."""

    article_id: int
    sentiment_label: str
    sentiment_score: int
    confidence: Optional[Decimal]


@dataclass(frozen=True)
class AnomalyAlert:
    """An anomaly detected in market data."""

    id: UUID
    symbol: str
    detected_at: datetime
    anomaly_type: str
    severity: Decimal
    description: str


@dataclass
class PortfolioPosition:
    """A single position within a portfolio."""

    symbol: str
    quantity: int
    purchase_price: Decimal
    purchased_at: date


@dataclass
class Portfolio:
    """A virtual portfolio tracking positions and capital."""

    id: UUID = field(default_factory=uuid4)
    risk_profile: RiskProfile = RiskProfile.MODERATE
    initial_capital: Decimal = Decimal("10000.00")
    cash_balance: Decimal = Decimal("10000.00")
    positions: list[PortfolioPosition] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class TradeRecommendation:
    """A recommendation produced by the decision agent."""

    symbol: str
    action: Recommendation
    confidence: Decimal
    reasoning: str
