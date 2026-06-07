"""API schemas for the standalone ML service."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=2, max_length=10, pattern=r"^[A-Z0-9]+$")
    horizon_days: int = Field(..., ge=1, le=5)


class PredictionItem(BaseModel):
    symbol: str
    target_date: date
    predicted_close: Decimal
    confidence_lower: Decimal
    confidence_upper: Decimal


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]


class VolumeItem(BaseModel):
    symbol: str
    target_date: date
    predicted_volume: int


class VolumeResponse(BaseModel):
    predictions: list[VolumeItem]


class LiquidityItem(BaseModel):
    symbol: str
    target_date: date
    prob_low: Decimal
    prob_medium: Decimal
    prob_high: Decimal
    predicted_tier: str


class LiquidityResponse(BaseModel):
    predictions: list[LiquidityItem]


class HealthResponse(BaseModel):
    status: str
    service: str