"""API schemas for the portfolio construction workflow."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RiskProfile = Literal["conservative", "moderate", "aggressive"]


class PortfolioOptimizationRequest(BaseModel):
    risk_profile: RiskProfile = "moderate"
    company_count: int = Field(default=5, ge=2, le=12)
    investment_amount: float = Field(default=10_000, gt=0, le=100_000_000)


class PortfolioAsset(BaseModel):
    symbol: str
    latest_price: float
    weight: float
    allocation_amount: float
    shares: int
    invested_amount: float
    beta: float
    covariance_with_market: float
    volatility: float
    capm_return: float


class PortfolioMetrics(BaseModel):
    expected_return: float
    volatility: float
    variance: float
    beta: float
    market_return: float
    risk_free_rate: float
    invested_amount: float
    cash_remaining: float


class EfficientFrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    is_pvm: bool = False


class PortfolioMethodology(BaseModel):
    observations: int
    trading_days_per_year: int
    market_variance: float
    market_risk_premium: float
    minimum_weight: float
    maximum_weight: float


class PortfolioOptimizationResponse(BaseModel):
    risk_profile: RiskProfile
    company_count: int
    investment_amount: float
    assets: list[PortfolioAsset]
    metrics: PortfolioMetrics
    efficient_frontier: list[EfficientFrontierPoint]
    methodology: PortfolioMethodology
    explanation: str
    explanation_source: Literal["openrouter", "lm_studio", "multi_agent", "fallback"]
    generated_at: datetime
    warnings: list[str] = []
