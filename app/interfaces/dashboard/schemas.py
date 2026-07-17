"""Schemas for the dashboard backend-for-frontend endpoint."""

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)


class CurrentMarketSnapshot(BaseModel):
    """Latest persisted OHLCV snapshot for the selected symbol."""

    date: date
    close: Decimal
    previous_close: Decimal
    volume: int
    average_volume: int


class DashboardBootstrapResponse(BaseModel):
    """Aggregated dashboard payload used by the frontend."""

    symbol: str
    current_snapshot: Optional[CurrentMarketSnapshot] = None
    historical_prices: list[HistoricalPriceItem] = Field(default_factory=list)
    price_predictions: list[PredictPriceItem] = Field(default_factory=list)
    sentiment: Optional[SentimentResponse] = None
    anomalies: list[AnomalyItem] = Field(default_factory=list)
    recommendation: Optional[RecommendationResponse] = None
    warnings: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    """Latest market row used by watchlists, markets, and screeners."""

    symbol: str
    date: date
    close: Decimal
    previous_close: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    average_volume: int
    sentiment_score: Decimal = Decimal("0")
    recommendation: Optional[str] = None
    recommendation_confidence: Decimal = Decimal("0")
    anomaly_count: int = 0


class MarketUniverseResponse(BaseModel):
    """Ranked set of actively traded BVMT instruments."""

    markets: list[MarketSnapshot] = Field(default_factory=list)
