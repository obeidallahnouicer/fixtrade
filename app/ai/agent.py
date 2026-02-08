"""
Main Decision Agent - Orchestrates AI Trading Assistant.

The DecisionAgent is the central coordinator that:
1. Manages user profiles and portfolio simulation
2. Generates daily recommendations
3. Executes automated trading decisions
4. Tracks performance metrics
5. Provides explainable AI reasoning

This is the primary interface for the AI trading assistant.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.config import ai_settings
from app.ai.profile import RiskProfile, UserProfileManager
from app.ai.portfolio import PortfolioManager
from app.ai.recommendations import RecommendationEngine, Recommendation
from app.ai.metrics import MetricsCalculator, PortfolioMetrics
from app.ai.explainability import ExplanationGenerator

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    AI Decision Agent for Portfolio Management.
    
    Main responsibilities:
    - Generate personalized recommendations
    - Manage virtual portfolio
    - Execute trades with risk management
    - Track and explain performance
    - Adapt to user risk profile
    
    Usage:
        agent = DecisionAgent(risk_profile="moderate")
        recommendations = await agent.get_daily_recommendations(session, top_n=5)
        
        result = await agent.execute_trade(
            session=session,
            symbol="AMEN",
            action="buy",
            quantity=10,
            price=12.50
        )
    """
    
    def __init__(
        self,
        portfolio_id: Optional[UUID] = None,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        initial_capital: float = ai_settings.default_initial_capital
    ):
        """
        Initialize Decision Agent.
        
        Args:
            portfolio_id: Existing portfolio ID (None creates new)
            risk_profile: Investment risk profile
            initial_capital: Starting capital in TND
        """
        self.risk_profile = risk_profile
        
        # Initialize components
        self.profile_manager = UserProfileManager(default_profile=risk_profile)
        self.portfolio = PortfolioManager(
            portfolio_id=portfolio_id,
            risk_profile=risk_profile,
            initial_capital=initial_capital
        )
        self.recommendation_engine = RecommendationEngine(risk_profile=risk_profile)
        self.metrics_calculator = MetricsCalculator()
        self.explainer = ExplanationGenerator()
        
        logger.info(
            f"DecisionAgent initialized: "
            f"profile={risk_profile}, capital={initial_capital} TND"
        )
    
    async def get_daily_recommendations(
        self,
        session: AsyncSession,
        top_n: int = 10,
        symbols: Optional[List[str]] = None,
        prediction_service = None
    ) -> List[Recommendation]:
        """
        Get daily trading recommendations.
        
        Args:
            session: Database session
            top_n: Number of recommendations
            symbols: Specific symbols to analyze (None = auto-select)
            prediction_service: Prediction inference service
        
        Returns:
            List of Recommendation objects
        """
        logger.info(f"Generating daily recommendations (top {top_n})")
        
        try:
            recommendations = await self.recommendation_engine.generate_recommendations(
                session=session,
                top_n=top_n,
                symbols=symbols,
                prediction_service=prediction_service
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def explain_recommendation(
        self,
        symbol: str,
        session: AsyncSession,
        prediction_service = None
    ) -> Optional[str]:
        """
        Get detailed explanation for a specific stock.
        
        Args:
            symbol: Stock symbol
            session: Database session
            prediction_service: Prediction service
        
        Returns:
            Natural language explanation
        """
        logger.info(f"Generating explanation for {symbol}")
        
        try:
            # Get portfolio context
            portfolio_context = {
                "total_value": self.portfolio.total_value,
                "cash_balance": self.portfolio.cash_balance,
                "equity_allocation": self.portfolio.equity_allocation,
                "position_count": len(self.portfolio.positions)
            }
            
            recommendation = await self.recommendation_engine.get_recommendation_with_explanation(
                symbol=symbol,
                session=session,
                prediction_service=prediction_service,
                user_context=portfolio_context
            )
            
            if recommendation:
                return recommendation.explanation
            
            return None
            
        except Exception as e:
            logger.error(f"Error explaining recommendation for {symbol}: {e}")
            return None
    
    async def execute_trade(
        self,
        session: AsyncSession,
        symbol: str,
        action: str,  # "buy" or "sell"
        quantity: int,
        price: float,
        generate_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a trade with risk management checks.
        
        Args:
            session: Database session
            symbol: Stock symbol
            action: "buy" or "sell"
            quantity: Number of shares
            price: Execution price
            generate_explanation: Whether to generate explanation
        
        Returns:
            Trade result dictionary
        """
        logger.info(f"Executing {action.upper()} {quantity} {symbol} @ {price}")
        
        result = {
            "success": False,
            "message": "",
            "explanation": None,
            "portfolio_value": self.portfolio.total_value
        }
        
        try:
            if action.lower() == "buy":
                success, message = self.portfolio.buy(symbol, quantity, price)
                result["success"] = success
                result["message"] = message
                
            elif action.lower() == "sell":
                success, message, pnl = self.portfolio.sell(symbol, quantity, price)
                result["success"] = success
                result["message"] = message
                result["profit_loss"] = pnl
            
            else:
                result["message"] = f"Action invalide: {action}"
                return result
            
            # Generate explanation if requested
            if result["success"] and generate_explanation:
                portfolio_context = self.portfolio.to_dict()
                explanation = await self.explainer.explain_portfolio_action(
                    action=action.lower(),
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    portfolio_context=portfolio_context
                )
                result["explanation"] = explanation
            
            result["portfolio_value"] = self.portfolio.total_value
            
            # Check stop losses after trade
            if result["success"]:
                triggered = self.portfolio.check_stop_losses()
                if triggered:
                    result["stop_loss_triggered"] = triggered
            
            logger.info(f"Trade executed: {result['message']}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            result["message"] = f"Erreur: {str(e)}"
        
        return result
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current market prices for portfolio positions.
        
        Args:
            prices: Dict mapping symbol -> current_price
        """
        self.portfolio.update_prices(prices)
        logger.debug(f"Updated prices for {len(prices)} symbols")
    
    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Portfolio snapshot dictionary
        """
        snapshot = self.portfolio.get_snapshot()
        
        return {
            "portfolio_id": str(snapshot.portfolio_id),
            "timestamp": snapshot.timestamp.isoformat(),
            "total_value": snapshot.total_value,
            "cash_balance": snapshot.cash_balance,
            "equity_value": snapshot.equity_value,
            "equity_allocation": self.portfolio.equity_allocation,
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "purchase_price": pos.purchase_price,
                    "current_price": pos.current_price,
                    "current_value": pos.current_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct
                }
                for pos in snapshot.positions
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dictionary of portfolio metrics
        """
        metrics = self.portfolio.calculate_metrics()
        
        return {
            "total_value": metrics.total_value,
            "total_return": metrics.total_return,
            "roi": metrics.roi,
            "volatility": metrics.volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "avg_gain": metrics.avg_gain,
            "avg_loss": metrics.avg_loss,
            "profit_factor": metrics.profit_factor,
            "days_active": metrics.days_active,
            "annualized_return": metrics.annualized_return
        }
    
    async def explain_performance(self) -> str:
        """
        Get natural language explanation of portfolio performance.
        
        Returns:
            Performance explanation
        """
        metrics = self.get_performance_metrics()
        explanation = await self.explainer.explain_metrics(metrics)
        return explanation
    
    def check_and_handle_stop_losses(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Check stop-loss triggers and execute sells.
        
        Args:
            current_prices: Current market prices
        
        Returns:
            List of executed stop-loss trades
        """
        # Update prices first
        self.update_market_prices(current_prices)
        
        # Check triggers
        triggered_symbols = self.portfolio.check_stop_losses()
        
        executed_trades = []
        
        for symbol in triggered_symbols:
            position = self.portfolio.positions.get(symbol)
            if position and position.current_price:
                # Execute stop-loss sell
                success, message, pnl = self.portfolio.sell(
                    symbol=symbol,
                    quantity=position.quantity,
                    price=position.current_price
                )
                
                if success:
                    executed_trades.append({
                        "symbol": symbol,
                        "quantity": position.quantity,
                        "price": position.current_price,
                        "profit_loss": pnl,
                        "reason": "stop_loss"
                    })
                    logger.info(f"Stop-loss executed for {symbol}: {message}")
        
        return executed_trades
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Position data or None
        """
        position = self.portfolio.positions.get(symbol)
        
        if position:
            return {
                "symbol": position.symbol,
                "quantity": position.quantity,
                "purchase_price": position.purchase_price,
                "current_price": position.current_price,
                "current_value": position.current_value,
                "unrealized_pnl": position.unrealized_pnl,
                "unrealized_pnl_pct": position.unrealized_pnl_pct,
                "purchased_at": position.purchased_at.isoformat()
            }
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize agent state.
        
        Returns:
            Complete agent state as dictionary
        """
        return {
            "risk_profile": self.risk_profile.value,
            "portfolio": self.portfolio.to_dict(),
            "metrics": self.get_performance_metrics()
        }
