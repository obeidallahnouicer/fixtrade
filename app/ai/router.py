"""
FastAPI Router for AI Decision Agent.

Exposes endpoints for:
- User profile management
- Daily recommendations
- Portfolio management
- Trade execution
- Performance metrics
- Explainability

Integrates with existing FastAPI application.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agent import DecisionAgent
from app.ai.profile import RiskProfile
from app.ai.config import ai_settings

logger = logging.getLogger(__name__)


# ── Pydantic Models ──────────────────────────────────────────────────

class ProfileQuestionnaireRequest(BaseModel):
    """User profile questionnaire."""
    age: int = Field(..., ge=18, le=100)
    investment_horizon: int = Field(..., ge=1, le=30, description="Years")
    income_stability: str = Field(..., pattern="^(high|medium|low)$")
    investment_experience: str = Field(..., pattern="^(beginner|intermediate|expert)$")
    loss_tolerance: int = Field(..., ge=1, le=5, description="1=Very Low, 5=Very High")
    financial_goals: str = Field(..., pattern="^(preservation|growth|aggressive_growth)$")


class ProfileResponse(BaseModel):
    """User profile response."""
    recommended_profile: str
    characteristics: Dict[str, Any]


class CreatePortfolioRequest(BaseModel):
    """Create portfolio request."""
    risk_profile: str = Field(default="moderate", pattern="^(conservative|moderate|aggressive)$")
    initial_capital: float = Field(default=10000.0, ge=1000.0)


class TradeRequest(BaseModel):
    """Execute trade request."""
    symbol: str = Field(..., min_length=1, max_length=20)
    action: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., ge=1)
    price: float = Field(..., gt=0)
    generate_explanation: bool = Field(default=True)


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    symbol: str
    signal: str
    strength: str
    explanation: str
    predicted_return: Optional[float]
    confidence: Optional[float]
    current_price: Optional[float]
    timestamp: str


# ── Router ───────────────────────────────────────────────────────────

router = APIRouter(prefix="/ai", tags=["AI Agent"])


# In-memory agent storage (replace with proper session management)
_agents: Dict[str, DecisionAgent] = {}


def get_or_create_agent(
    portfolio_id: str = "default",
    risk_profile: str = "moderate"
) -> DecisionAgent:
    """Get or create decision agent."""
    if portfolio_id not in _agents:
        _agents[portfolio_id] = DecisionAgent(
            risk_profile=RiskProfile(risk_profile),
            initial_capital=ai_settings.default_initial_capital
        )
    return _agents[portfolio_id]


# ── Profile Endpoints ────────────────────────────────────────────────

@router.post("/profile/questionnaire", response_model=ProfileResponse)
async def recommend_profile(request: ProfileQuestionnaireRequest):
    """
    Recommend risk profile based on questionnaire.
    
    Returns recommended profile and its characteristics.
    """
    from app.ai.profile import UserProfileManager
    
    try:
        manager = UserProfileManager()
        
        questionnaire = {
            "age": request.age,
            "investment_horizon": request.investment_horizon,
            "income_stability": request.income_stability,
            "investment_experience": request.investment_experience,
            "loss_tolerance": request.loss_tolerance,
            "financial_goals": request.financial_goals
        }
        
        recommended = manager.recommend_profile(questionnaire)
        characteristics = manager.get_characteristics(recommended)
        
        return {
            "recommended_profile": recommended.value,
            "characteristics": {
                "max_position_size": characteristics.max_position_size,
                "max_equity_allocation": characteristics.max_equity_allocation,
                "stop_loss_threshold": characteristics.stop_loss_threshold,
                "min_holding_days": characteristics.min_holding_days,
                "preferred_liquidity": characteristics.preferred_liquidity,
                "description": characteristics.description
            }
        }
        
    except Exception as e:
        logger.error(f"Profile recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Portfolio Endpoints ──────────────────────────────────────────────

@router.post("/portfolio/create")
async def create_portfolio(request: CreatePortfolioRequest):
    """
    Create new portfolio with risk profile.
    
    Returns portfolio ID and initial state.
    """
    try:
        agent = DecisionAgent(
            risk_profile=RiskProfile(request.risk_profile),
            initial_capital=request.initial_capital
        )
        
        portfolio_id = str(agent.portfolio.portfolio_id)
        _agents[portfolio_id] = agent
        
        return {
            "portfolio_id": portfolio_id,
            "risk_profile": request.risk_profile,
            "initial_capital": request.initial_capital,
            "snapshot": agent.get_portfolio_snapshot()
        }
        
    except Exception as e:
        logger.error(f"Portfolio creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/{portfolio_id}/snapshot")
async def get_portfolio_snapshot(portfolio_id: str = "default"):
    """Get current portfolio snapshot."""
    agent = get_or_create_agent(portfolio_id)
    return agent.get_portfolio_snapshot()


@router.get("/portfolio/{portfolio_id}/performance")
async def get_portfolio_performance(portfolio_id: str = "default"):
    """Get portfolio performance metrics."""
    agent = get_or_create_agent(portfolio_id)
    return agent.get_performance_metrics()


@router.get("/portfolio/{portfolio_id}/performance/explain")
async def explain_performance(portfolio_id: str = "default"):
    """Get natural language explanation of performance."""
    agent = get_or_create_agent(portfolio_id)
    explanation = await agent.explain_performance()
    return {"explanation": explanation}


@router.get("/portfolio/{portfolio_id}/position/{symbol}")
async def get_position(portfolio_id: str, symbol: str):
    """Get position details for a symbol."""
    agent = get_or_create_agent(portfolio_id)
    position = agent.get_position(symbol)
    
    if position is None:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    
    return position


# ── Recommendation Endpoints ─────────────────────────────────────────

@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_daily_recommendations(
    portfolio_id: str = Query(default="default"),
    top_n: int = Query(default=10, ge=1, le=50),
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols")
):
    """
    Get daily trading recommendations.
    
    Args:
        portfolio_id: Portfolio identifier
        top_n: Number of recommendations (1-50)
        symbols: Optional comma-separated list of symbols to analyze
    
    Returns:
        List of recommendations with explanations
    """
    agent = get_or_create_agent(portfolio_id)
    
    # Parse symbols if provided
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # TODO: Get database session
    # TODO: Get prediction service
    # For now, return empty list with placeholder
    
    return []


@router.get("/recommendations/{symbol}/explain")
async def explain_recommendation(
    symbol: str,
    portfolio_id: str = Query(default="default")
):
    """
    Get detailed explanation for a specific stock recommendation.
    
    Uses Groq AI to generate natural language explanation.
    """
    agent = get_or_create_agent(portfolio_id)
    
    # TODO: Get database session and prediction service
    
    return {
        "symbol": symbol,
        "explanation": "Explanation endpoint - requires database session integration"
    }


# ── Trade Execution Endpoints ────────────────────────────────────────

@router.post("/portfolio/{portfolio_id}/trade")
async def execute_trade(portfolio_id: str, request: TradeRequest):
    """
    Execute a trade (buy or sell).
    
    Includes risk management checks and optional explanation generation.
    """
    agent = get_or_create_agent(portfolio_id)
    
    try:
        # TODO: Get database session for explanation
        result = await agent.execute_trade(
            session=None,  # Placeholder
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            price=request.price,
            generate_explanation=request.generate_explanation
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/{portfolio_id}/prices/update")
async def update_prices(portfolio_id: str, prices: Dict[str, float]):
    """
    Update current market prices for portfolio positions.
    
    Request body: {"SYMBOL": price, ...}
    """
    agent = get_or_create_agent(portfolio_id)
    agent.update_market_prices(prices)
    
    return {
        "updated": len(prices),
        "portfolio_value": agent.portfolio.total_value
    }


@router.post("/portfolio/{portfolio_id}/stop-loss/check")
async def check_stop_losses(portfolio_id: str, prices: Dict[str, float]):
    """
    Check and execute stop-loss orders.
    
    Returns list of executed stop-loss trades.
    """
    agent = get_or_create_agent(portfolio_id)
    executed = agent.check_and_handle_stop_losses(prices)
    
    return {
        "triggered": len(executed),
        "trades": executed,
        "portfolio_value": agent.portfolio.total_value
    }


# ── Status Endpoint ──────────────────────────────────────────────────

@router.get("/status")
async def get_ai_status():
    """Get AI module status and configuration."""
    return {
        "module": "AI Decision Agent",
        "groq_configured": bool(ai_settings.groq_api_key),
        "groq_model": ai_settings.groq_model,
        "default_capital": ai_settings.default_initial_capital,
        "active_portfolios": len(_agents),
        "risk_profiles": ["conservative", "moderate", "aggressive"]
    }
