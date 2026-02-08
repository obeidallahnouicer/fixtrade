"""
Pydantic schemas for trading API request/response validation.

These schemas enforce input validation and define the API contract.
All fields use strict typing with constraints.
No business logic belongs here.
"""

from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field

SYMBOL_DESCRIPTION = "BVMT stock ticker symbol"
SYMBOL_PATTERN = r"^[A-Z0-9]+$"
SYMBOL_MIN_LEN = 2
SYMBOL_MAX_LEN = 10


class PredictPriceRequest(BaseModel):
    """Request schema for price prediction endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
        horizon_days: Number of future days to predict (1-5).
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )
    horizon_days: int = Field(
        ..., ge=1, le=5, description="Prediction horizon in trading days (1-5)"
    )


class PredictPriceItem(BaseModel):
    """A single predicted price point in the response."""

    symbol: str
    target_date: date
    predicted_close: Decimal
    confidence_lower: Decimal
    confidence_upper: Decimal


class PredictPriceResponse(BaseModel):
    """Response schema for price prediction endpoint."""

    predictions: list[PredictPriceItem]


class GetSentimentRequest(BaseModel):
    """Request schema for sentiment analysis endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
        target_date: Optional date for the sentiment query.
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )
    target_date: date | None = Field(
        default=None, description="Target date for sentiment query"
    )


class SentimentResponse(BaseModel):
    """Response schema for sentiment analysis endpoint."""

    symbol: str
    date: date
    score: Decimal
    sentiment: str
    article_count: int


class DetectAnomaliesRequest(BaseModel):
    """Request schema for anomaly detection endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )


class AnomalyItem(BaseModel):
    """A single anomaly in the response."""

    id: UUID
    symbol: str
    anomaly_type: str
    severity: Decimal
    description: str


class DetectAnomaliesResponse(BaseModel):
    """Response schema for anomaly detection endpoint."""

    anomalies: list[AnomalyItem]


class GetRecommendationRequest(BaseModel):
    """Request schema for trade recommendation endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
        portfolio_id: UUID of the portfolio to evaluate.
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )
    portfolio_id: UUID = Field(
        ..., description="UUID of the portfolio"
    )


class RecommendationResponse(BaseModel):
    """Response schema for trade recommendation endpoint."""

    symbol: str
    action: str
    confidence: Decimal
    reasoning: str


# ------------------------------------------------------------------
# Volume Prediction schemas
# ------------------------------------------------------------------


class PredictVolumeRequest(BaseModel):
    """Request schema for volume prediction endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
        horizon_days: Number of future days to predict (1-5).
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )
    horizon_days: int = Field(
        ..., ge=1, le=5, description="Prediction horizon in trading days (1-5)"
    )


class PredictVolumeItem(BaseModel):
    """A single predicted volume point in the response."""

    symbol: str
    target_date: date
    predicted_volume: int


class PredictVolumeResponse(BaseModel):
    """Response schema for volume prediction endpoint."""

    predictions: list[PredictVolumeItem]


# ------------------------------------------------------------------
# Liquidity Probability schemas
# ------------------------------------------------------------------


class PredictLiquidityRequest(BaseModel):
    """Request schema for liquidity probability endpoint.

    Attributes:
        symbol: BVMT stock ticker (2-10 uppercase chars).
        horizon_days: Number of future days to predict (1-5).
    """

    symbol: str = Field(
        ...,
        min_length=SYMBOL_MIN_LEN,
        max_length=SYMBOL_MAX_LEN,
        pattern=SYMBOL_PATTERN,
        description=SYMBOL_DESCRIPTION,
    )
    horizon_days: int = Field(
        ..., ge=1, le=5, description="Prediction horizon in trading days (1-5)"
    )


class PredictLiquidityItem(BaseModel):
    """A single liquidity probability forecast in the response."""

    symbol: str
    target_date: date
    prob_low: Decimal
    prob_medium: Decimal
    prob_high: Decimal
    predicted_tier: str


class PredictLiquidityResponse(BaseModel):
    """Response schema for liquidity probability endpoint."""

    forecasts: list[PredictLiquidityItem]


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str
    version: str


class ErrorResponse(BaseModel):
    """Standard error response returned by all error handlers."""

    error: str
    detail: str | None = None
