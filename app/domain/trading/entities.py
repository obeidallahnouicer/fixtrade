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
class SentimentScore:
    """Aggregated daily sentiment score for a given stock."""

    symbol: str
    date: date
    score: Decimal
    sentiment: Sentiment
    article_count: int


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
