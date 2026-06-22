"""Schemas for the dashboard backend-for-frontend endpoint."""

from typing import Optional

from pydantic import BaseModel, Field

from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)


class DashboardBootstrapResponse(BaseModel):
    """Aggregated dashboard payload used by the frontend."""

    symbol: str
    historical_prices: list[HistoricalPriceItem] = Field(default_factory=list)
    price_predictions: list[PredictPriceItem] = Field(default_factory=list)
    sentiment: Optional[SentimentResponse] = None
    anomalies: list[AnomalyItem] = Field(default_factory=list)
    recommendation: Optional[RecommendationResponse] = None
    warnings: list[str] = Field(default_factory=list)