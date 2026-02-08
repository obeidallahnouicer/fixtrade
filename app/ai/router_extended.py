"""
Extended API Router for Portfolio Optimization & Simulation.

New endpoints:
- /ai/portfolio/optimize - Run MPT optimization
- /ai/portfolio/efficient-frontier - Get efficient frontier
- /ai/portfolio/recommendations/detailed - Get CAPM-based recommendations  
- /ai/portfolio/simulate - Run backtest simulation
- /ai/portfolio/explain - Get portfolio-level explanation
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.optimization import PortfolioOptimizer, CAPMCalculator
from app.ai.decision_engine import DecisionEngine
from app.ai.simulator import PortfolioSimulator
from app.ai.llm_explainer import LLMExplainer, ExplanationContext, LLMConfig
from app.ai.profile import RiskProfile
from app.ai.config import ai_settings
from app.ai.data_service import PortfolioDataService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/portfolio", tags=["Portfolio Optimization"])


# Database dependency
async def get_db() -> AsyncSession:
    """Get database session."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as AS
    from sqlalchemy.orm import sessionmaker
    
    engine = create_async_engine(
        settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
        echo=False
    )
    
    async_session = sessionmaker(
        engine, class_=AS, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


# ── Pydantic Models ──────────────────────────────────────────────────

RISK_PROFILE_PATTERN = "^(conservative|moderate|aggressive)$"


class LLMConfigRequest(BaseModel):
    """LLM configuration from frontend."""
    provider: str = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1, max_length=100)
    api_key: str = Field(..., min_length=1, max_length=500)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=50, le=1000)
    enable_reasoning: bool = Field(default=False)

class OptimizationRequest(BaseModel):
    """Portfolio optimization request."""
    symbols: List[str] = Field(..., min_length=2, max_length=50)
    risk_profile: str = Field(default="moderate", pattern=RISK_PROFILE_PATTERN)
    optimization_method: str = Field(default="max_sharpe", pattern="^(min_variance|max_sharpe)$")
    portfolio_id: Optional[str] = None
    user_id: Optional[str] = None


class EfficientFrontierRequest(BaseModel):
    """Efficient frontier request."""
    symbols: List[str] = Field(..., min_length=2, max_length=50)
    num_points: int = Field(default=50, ge=10, le=200)


class RecommendationRequest(BaseModel):
    """CAPM-based recommendation request."""
    symbols: Optional[List[str]] = Field(default=None, min_length=1, max_length=100)
    risk_profile: str = Field(default="moderate", pattern=RISK_PROFILE_PATTERN)
    portfolio_id: Optional[str] = None
    user_id: Optional[str] = None
    llm_config: Optional[LLMConfigRequest] = None
    top_n: int = Field(default=10, ge=1, le=50)


class SimulationRequest(BaseModel):
    """Portfolio simulation request."""
    symbols: List[str] = Field(..., min_length=2, max_length=50)
    risk_profile: str = Field(default="moderate", pattern=RISK_PROFILE_PATTERN)
    initial_capital: float = Field(default=10000.0, ge=1000.0)
    rebalance_frequency: str = Field(default="weekly", pattern="^(daily|weekly|monthly)$")
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class OptimizationResponse(BaseModel):
    """Optimization result."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    method: str
    risk_profile: str


class RecommendationDetailedResponse(BaseModel):
    """Detailed recommendation with CAPM metrics."""
    symbol: str
    action: str
    confidence: float
    current_weight: float
    target_weight: float
    expected_return: float
    beta: float
    risk_contribution: float
    anomaly_detected: bool
    explanation: str


class SimulationResponse(BaseModel):
    """Simulation result with metrics."""
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    dates: List[str]
    value_history: List[float]


class EfficientFrontierResponse(BaseModel):
    """Efficient frontier data."""
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    weights: List[Dict[str, float]]


class PortfolioExplanationResponse(BaseModel):
    """Portfolio-level explanation."""
    summary: str
    risk_profile: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    recommendations_count: Dict[str, int]


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Run portfolio optimization using Modern Portfolio Theory.
    
    Supports:
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Risk profile constraints
    
    Returns optimized weights and metrics.
    """
    try:
        # TODO: Fetch real historical returns from database
        # Placeholder: generate synthetic returns
        import numpy as np
        num_symbols = len(request.symbols)
        returns = np.random.randn(250, num_symbols) * 0.02 + 0.001
        
        optimizer = PortfolioOptimizer(
            returns=returns,
            symbols=request.symbols,
            risk_profile=RiskProfile(request.risk_profile)
        )
        
        if request.optimization_method == "min_variance":
            result = optimizer.minimum_variance_portfolio()
        else:
            result = optimizer.maximum_sharpe_portfolio()
        
        return OptimizationResponse(
            weights=result.weights,
            expected_return=result.expected_return * 100,
            volatility=result.volatility * 100,
            sharpe_ratio=result.sharpe_ratio,
            diversification_ratio=result.diversification_ratio,
            method=request.optimization_method,
            risk_profile=request.risk_profile
        )
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/efficient-frontier", response_model=EfficientFrontierResponse)
async def get_efficient_frontier(request: EfficientFrontierRequest):
    """
    Calculate efficient frontier for asset universe.
    
    Returns frontier points: returns, volatilities, Sharpe ratios, weights.
    """
    try:
        # TODO: Fetch real returns from database
        import numpy as np
        num_symbols = len(request.symbols)
        returns = np.random.randn(250, num_symbols) * 0.02 + 0.001
        
        optimizer = PortfolioOptimizer(
            returns=returns,
            symbols=request.symbols
        )
        
        frontier_results = optimizer.efficient_frontier(num_points=request.num_points)
        
        return EfficientFrontierResponse(
            returns=[r.expected_return * 100 for r in frontier_results],
            volatilities=[r.volatility * 100 for r in frontier_results],
            sharpe_ratios=[r.sharpe_ratio for r in frontier_results],
            weights=[r.weights for r in frontier_results]
        )
    
    except Exception as e:
        logger.error(f"Efficient frontier calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/detailed", response_model=List[RecommendationDetailedResponse])
async def get_detailed_recommendations(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate CAPM-based BUY/SELL/HOLD recommendations.
    
    Fully wired with database:
    - Fetches historical returns
    - Gets market returns (TUNINDEX)
    - Integrates anomaly detection
    - Retrieves current portfolio weights
    - Supports dynamic LLM configuration from frontend
    
    Returns detailed recommendations with natural language explanations.
    """
    try:
        data_service = PortfolioDataService()
        
        # Fetch symbols if not provided
        if not request.symbols:
            # Get top liquid stocks
            query = """
                SELECT DISTINCT symbol
                FROM stock_prices
                WHERE seance >= CURRENT_DATE - INTERVAL '30 days'
                  AND quantite_negociee > 0
                GROUP BY symbol
                ORDER BY AVG(quantite_negociee) DESC
                LIMIT :limit
            """
            from sqlalchemy import text
            result = await db.execute(text(query), {"limit": request.top_n * 2})
            request.symbols = [row[0] for row in result.fetchall()]
        
        # Fetch historical returns
        returns, available_symbols = await data_service.fetch_historical_returns(
            db, request.symbols, lookback_days=250
        )
        
        # Fetch market returns
        market_returns = await data_service.fetch_market_returns(db, lookback_days=250)
        
        # Ensure market_returns matches returns length
        min_len = min(len(returns), len(market_returns))
        returns = returns[-min_len:]
        market_returns = market_returns[-min_len:]
        
        # Fetch anomalies
        anomalies = await data_service.fetch_anomaly_status(db, available_symbols)
        
        # Fetch current weights
        current_weights = await data_service.fetch_current_weights(
            db,
            portfolio_id=request.portfolio_id,
            user_id=request.user_id
        )
        
        # Create LLM config if provided
        llm_config = None
        use_llm = False
        if request.llm_config:
            llm_config = LLMConfig(
                provider=request.llm_config.provider,
                model=request.llm_config.model,
                api_key=request.llm_config.api_key,
                temperature=request.llm_config.temperature,
                max_tokens=request.llm_config.max_tokens,
                enable_reasoning=request.llm_config.enable_reasoning
            )
            use_llm = True
        
        # Generate recommendations
        engine = DecisionEngine(
            risk_profile=RiskProfile(request.risk_profile),
            use_llm_explanation=use_llm,
            llm_config=llm_config
        )
        
        portfolio_recommendation = engine.generate_recommendations(
            returns=returns,
            market_returns=market_returns,
            current_portfolio=current_weights,
            risk_profile=RiskProfile(request.risk_profile),
            anomalies=anomalies,
            total_value=10000.0,  # TODO: Get from user portfolio
            cash_available=0.0     # TODO: Get from user portfolio
        )
        
        # Extract signals and convert to dict format
        signals = portfolio_recommendation.signals
        
        # Sort by confidence and take top N
        signals.sort(key=lambda x: x.confidence, reverse=True)
        recommendations = [
            {
                "symbol": s.symbol,
                "decision": s.decision.value,
                "confidence": s.confidence,
                "optimal_weight": s.optimal_weight,
                "current_weight": s.current_weight,
                "weight_delta": s.weight_delta,
                "expected_return": s.expected_return,
                "capm_return": s.capm_return,
                "contribution_to_risk": s.contribution_to_risk,
                "beta": s.beta,
                "anomaly_detected": s.anomaly_detected,
                "diversification_benefit": s.diversification_benefit,
                "reasons": s.reasons
            }
            for s in signals[:request.top_n]
        ]
        
        # Save to database if portfolio_id provided
        if request.portfolio_id:
            await data_service.save_recommendations(
                db, request.portfolio_id, recommendations
            )
        
        return [
            RecommendationDetailedResponse(
                symbol=rec["symbol"],
                action=rec["action"],
                confidence=rec["confidence"],
                current_weight=rec["current_weight"],
                target_weight=rec["target_weight"],
                expected_return=rec["expected_return"],
                beta=rec["beta"],
                risk_contribution=rec["risk_contribution"],
                anomaly_detected=rec["anomaly_detected"],
                explanation=rec["explanation"]
            )
            for rec in recommendations
        ]
    
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_portfolio(request: SimulationRequest):
    """
    Run portfolio backtest simulation.
    
    Features:
    - Daily/weekly/monthly rebalancing
    - Transaction costs
    - Performance metrics (Sharpe, Sortino, max drawdown)
    - Win rate & profit factor
    
    Returns complete simulation metrics and value history.
    """
    try:
        # TODO: Fetch historical prices from database
        # Placeholder: generate price history
        import numpy as np
        import pandas as pd
        
        num_days = 250
        start_price = 100.0
        
        price_history = {}
        for symbol in request.symbols:
            daily_returns = np.random.randn(num_days) * 0.02 + 0.001
            prices = start_price * np.cumprod(1 + daily_returns)
            price_history[symbol] = prices
        
        dates = pd.date_range(
            start=request.start_date or datetime.now().date(),
            periods=num_days,
            freq="D"
        ).date.tolist()
        
        simulator = PortfolioSimulator(
            initial_capital=request.initial_capital,
            commission_rate=0.001,
            risk_free_rate=ai_settings.risk_free_rate
        )
        
        # TODO: Implement full simulation loop with rebalancing
        # For MVP: calculate metrics from placeholder data
        
        metrics = simulator.calculate_metrics()
        
        return SimulationResponse(
            initial_capital=metrics.initial_capital,
            final_value=metrics.final_value,
            total_return_pct=metrics.total_return_pct,
            annualized_return=metrics.annualized_return,
            volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            dates=[d.isoformat() for d in metrics.dates],
            value_history=metrics.value_history
        )
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=PortfolioExplanationResponse)
async def explain_portfolio(
    recommendations: List[Dict[str, Any]] = Body(...),
    portfolio_metrics: Dict[str, float] = Body(...),
    risk_profile: str = Body(default="moderate"),
    use_llm: bool = Body(default=False)
):
    """
    Generate portfolio-level natural language explanation.
    
    Summarizes:
    - Overall strategy
    - Risk-return profile
    - Number of BUY/SELL/HOLD signals
    - Key metrics interpretation
    
    Can use LLM or template-based generation.
    """
    try:
        explainer = LLMExplainer(use_llm=use_llm)
        
        summary = explainer.explain_portfolio(
            recommendations=recommendations,
            portfolio_metrics=portfolio_metrics,
            risk_profile=risk_profile
        )
        
        buy_count = sum(1 for r in recommendations if r.get("action") == "BUY")
        sell_count = sum(1 for r in recommendations if r.get("action") == "SELL")
        hold_count = sum(1 for r in recommendations if r.get("action") == "HOLD")
        
        return PortfolioExplanationResponse(
            summary=summary,
            risk_profile=risk_profile,
            expected_return=portfolio_metrics.get("expected_return", 0.0),
            volatility=portfolio_metrics.get("volatility", 0.0),
            sharpe_ratio=portfolio_metrics.get("sharpe_ratio", 0.0),
            diversification_ratio=portfolio_metrics.get("diversification_ratio", 0.0),
            recommendations_count={
                "BUY": buy_count,
                "SELL": sell_count,
                "HOLD": hold_count
            }
        )
    
    except Exception as e:
        logger.error(f"Portfolio explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
