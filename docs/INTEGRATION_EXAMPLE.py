"""
Integration Example: Wiring Portfolio Optimization to Existing System

This shows how to connect the new modules with existing:
- Database (predictions, prices)
- Prediction service (anomaly detection)
- Existing agent.py
- Main FastAPI app
"""

import logging
from datetime import date, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.optimization import PortfolioOptimizer
from app.ai.decision_engine import DecisionEngine
from app.ai.simulator import PortfolioSimulator
from app.ai.llm_explainer import LLMExplainer
from app.ai.profile import RiskProfile

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Example 1: Full Pipeline - From Database to Recommendations
# ═══════════════════════════════════════════════════════════════════════

async def generate_portfolio_recommendations_full(
    db: AsyncSession,
    symbols: List[str],
    risk_profile: RiskProfile,
    lookback_days: int = 250
) -> List[Dict]:
    """
    Complete pipeline: DB → Optimization → Decisions → Explanations.
    
    Steps:
    1. Fetch historical prices from database
    2. Calculate returns
    3. Fetch market returns
    4. Get anomaly detection results
    5. Get current portfolio weights
    6. Run optimization
    7. Generate BUY/SELL/HOLD decisions
    8. Generate explanations
    
    This is what you'd call from the API endpoint.
    """
    
    # Step 1: Fetch historical prices from database
    # TODO: Replace with actual database query
    # Example query:
    # query = """
    #     SELECT symbol, date, close_price
    #     FROM stock_prices
    #     WHERE symbol = ANY(:symbols)
    #       AND date >= :start_date
    #     ORDER BY symbol, date
    # """
    # result = await db.execute(query, {
    #     "symbols": symbols,
    #     "start_date": date.today() - timedelta(days=lookback_days)
    # })
    
    # Placeholder: generate synthetic price data
    prices_df = pd.DataFrame()
    for symbol in symbols:
        dates = pd.date_range(
            end=date.today(),
            periods=lookback_days,
            freq="D"
        )
        base_price = 100.0
        returns = np.random.randn(lookback_days) * 0.02 + 0.001
        prices = base_price * np.cumprod(1 + returns)
        
        symbol_df = pd.DataFrame({
            "symbol": symbol,
            "date": dates,
            "close": prices
        })
        prices_df = pd.concat([prices_df, symbol_df], ignore_index=True)
    
    # Step 2: Calculate returns
    returns_data = []
    for symbol in symbols:
        symbol_prices = prices_df[prices_df["symbol"] == symbol]["close"].values
        symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
        returns_data.append(symbol_returns)
    
    returns = np.column_stack(returns_data)
    
    # Step 3: Fetch market returns (e.g., TUNINDEX)
    # TODO: Replace with actual market index query
    market_returns = np.random.randn(returns.shape[0]) * 0.015 + 0.0008
    
    # Step 4: Get anomaly detection results
    # TODO: Integrate with prediction service
    # Example:
    # from prediction.inference import PredictionService
    # prediction_service = PredictionService()
    # anomalies = await prediction_service.get_anomalies(symbols)
    
    anomalies = dict.fromkeys(symbols, False)
    
    # Step 5: Get current portfolio weights (if any)
    # TODO: Query from user's current portfolio
    # query = """
    #     SELECT symbol, quantity, current_value
    #     FROM portfolio_positions
    #     WHERE user_id = :user_id
    # """
    
    current_weights = {}
    
    # Step 6: Generate recommendations using DecisionEngine
    engine = DecisionEngine(
        risk_profile=risk_profile,
        use_llm_explanation=False  # Start with templates
    )
    
    recommendations = engine.generate_recommendations(
        symbols=symbols,
        returns=returns,
        market_returns=market_returns,
        current_weights=current_weights,
        anomalies=anomalies
    )
    
    logger.info(
        f"Generated {len(recommendations)} recommendations "
        f"for {risk_profile.value} profile"
    )
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════════
# Example 2: Backtest Simulation
# ═══════════════════════════════════════════════════════════════════════

async def run_portfolio_backtest(
    db: AsyncSession,
    symbols: List[str],
    risk_profile: RiskProfile,
    initial_capital: float = 10000.0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    rebalance_frequency: str = "weekly"
) -> Dict:
    """
    Run historical backtest with rebalancing.
    
    Returns performance metrics.
    """
    
    # Fetch historical prices
    # TODO: DB query
    
    # Initialize simulator
    simulator = PortfolioSimulator(
        initial_capital=initial_capital,
        commission_rate=0.001,
        risk_free_rate=0.05
    )
    
    # Get optimization engine
    # Simplified: would iterate through time, reoptimize, rebalance
    
    # For MVP: return placeholder metrics
    metrics = simulator.calculate_metrics()
    
    return {
        "initial_capital": metrics.initial_capital,
        "final_value": metrics.final_value,
        "total_return_pct": metrics.total_return_pct,
        "annualized_return": metrics.annualized_return,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate
    }


# ═══════════════════════════════════════════════════════════════════════
# Example 3: Integration with Existing Agent
# ═══════════════════════════════════════════════════════════════════════

class EnhancedDecisionAgent:
    """
    Extended version of existing DecisionAgent with optimization.
    
    Wraps the new modules while maintaining backward compatibility.
    """
    
    def __init__(
        self,
        risk_profile: RiskProfile,
        initial_capital: float = 10000.0,
        use_llm_explanation: bool = False
    ):
        self.risk_profile = risk_profile
        self.initial_capital = initial_capital
        
        # New components
        self.decision_engine = DecisionEngine(
            risk_profile=risk_profile,
            use_llm_explanation=use_llm_explanation
        )
        self.explainer = LLMExplainer(use_llm=use_llm_explanation)
        
        # Would also initialize existing Portfolio, etc.
    
    async def get_optimized_recommendations(
        self,
        db: AsyncSession,
        symbols: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get recommendations using new optimization engine.
        
        This replaces/enhances the existing get_daily_recommendations().
        """
        
        # 1. Get symbols (from DB or parameter)
        if symbols is None:
            # TODO: Query most liquid BVMT stocks
            symbols = await self._get_top_liquid_stocks(db, top_n=20)
        
        # 2. Fetch data
        returns, market_returns = await self._fetch_returns(db, symbols)
        anomalies = await self._fetch_anomalies(db, symbols)
        current_weights = await self._get_current_weights(db)
        
        # 3. Generate recommendations
        recommendations = self.decision_engine.generate_recommendations(
            symbols=symbols,
            returns=returns,
            market_returns=market_returns,
            current_weights=current_weights,
            anomalies=anomalies
        )
        
        # 4. Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 5. Return top N
        return recommendations[:top_n]
    
    async def _get_top_liquid_stocks(
        self,
        db: AsyncSession,
        top_n: int
    ) -> List[str]:
        """Get most liquid stocks from database."""
        # TODO: Query by average volume
        return ["STOCK1", "STOCK2", "STOCK3"]
    
    async def _fetch_returns(
        self,
        db: AsyncSession,
        symbols: List[str]
    ) -> tuple:
        """Fetch historical returns from database."""
        # TODO: DB query
        num_symbols = len(symbols)
        returns = np.random.randn(250, num_symbols) * 0.02 + 0.001
        market_returns = np.random.randn(250) * 0.015 + 0.0008
        return returns, market_returns
    
    async def _fetch_anomalies(
        self,
        db: AsyncSession,
        symbols: List[str]
    ) -> Dict[str, bool]:
        """Get anomaly detection results."""
        # TODO: Query predictions table or call prediction service
        return dict.fromkeys(symbols, False)
    
    async def _get_current_weights(
        self,
        db: AsyncSession
    ) -> Dict[str, float]:
        """Get current portfolio weights."""
        # TODO: Query user's portfolio
        return {}


# ═══════════════════════════════════════════════════════════════════════
# Example 4: Registering Extended Router
# ═══════════════════════════════════════════════════════════════════════

"""
In app/main.py:

from app.ai.router import router as ai_router
from app.ai.router_extended import router as portfolio_router

app = FastAPI()

# Existing endpoints
app.include_router(ai_router)

# New optimization endpoints
app.include_router(portfolio_router)
"""


# ═══════════════════════════════════════════════════════════════════════
# Example 5: Database Query Helpers
# ═══════════════════════════════════════════════════════════════════════

async def fetch_historical_returns(
    db: AsyncSession,
    symbols: List[str],
    lookback_days: int = 250
) -> np.ndarray:
    """
    Fetch historical returns from database.
    
    Returns: Array of shape (days, num_symbols)
    """
    query = """
    WITH daily_prices AS (
        SELECT 
            symbol,
            date,
            close as price,
            LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_price
        FROM stock_prices
        WHERE symbol = ANY(:symbols)
          AND date >= :start_date
        ORDER BY date
    )
    SELECT 
        symbol,
        date,
        (price - prev_price) / prev_price as return
    FROM daily_prices
    WHERE prev_price IS NOT NULL
    """
    
    # TODO: Execute query and reshape to matrix
    # result = await db.execute(query, {...})
    # returns_df = pd.DataFrame(result.fetchall())
    # returns_matrix = returns_df.pivot(index='date', columns='symbol', values='return')
    # return returns_matrix.values
    
    # Placeholder
    num_symbols = len(symbols)
    return np.random.randn(lookback_days, num_symbols) * 0.02 + 0.001


async def fetch_market_returns(
    db: AsyncSession,
    lookback_days: int = 250
) -> np.ndarray:
    """
    Fetch market index (TUNINDEX) returns.
    
    Returns: Array of shape (days,)
    """
    query = """
    SELECT 
        date,
        (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) as return
    FROM market_index
    WHERE date >= :start_date
    ORDER BY date
    """
    
    # TODO: Execute query
    # Placeholder
    return np.random.randn(lookback_days) * 0.015 + 0.0008


async def fetch_anomaly_status(
    db: AsyncSession,
    symbols: List[str]
) -> Dict[str, bool]:
    """
    Get latest anomaly detection status for symbols.
    
    Returns: Dict of symbol -> is_anomaly
    """
    query = """
    SELECT DISTINCT ON (symbol)
        symbol,
        has_anomaly
    FROM predictions
    WHERE symbol = ANY(:symbols)
    ORDER BY symbol, prediction_date DESC
    """
    
    # TODO: Execute query
    # result = await db.execute(query, {"symbols": symbols})
    # return dict(result.fetchall())
    
    # Placeholder
    return dict.fromkeys(symbols, False)


# ═══════════════════════════════════════════════════════════════════════
# Example 6: Testing the Integration
# ═══════════════════════════════════════════════════════════════════════

async def test_full_pipeline():
    """
    Test the full pipeline with synthetic data.
    
    Use this to validate integration before DB connection.
    """
    
    symbols = ["BNA", "STB", "SOTUMAG", "SOTRAPIL", "BIAT"]
    
    # Generate synthetic data
    returns = np.random.randn(250, len(symbols)) * 0.02 + 0.001
    market_returns = np.random.randn(250) * 0.015 + 0.0008
    anomalies = dict.fromkeys(symbols, False)
    
    # Test optimization
    optimizer = PortfolioOptimizer(
        returns=returns,
        symbols=symbols,
        risk_profile=RiskProfile.MODERATE
    )
    
    opt_result = optimizer.maximum_sharpe_portfolio()
    print(f"Optimal Portfolio Sharpe: {opt_result.sharpe_ratio:.2f}")
    print(f"Weights: {opt_result.weights}")
    
    # Test decision engine
    engine = DecisionEngine(
        risk_profile=RiskProfile.MODERATE,
        use_llm_explanation=False
    )
    
    recommendations = engine.generate_recommendations(
        symbols=symbols,
        returns=returns,
        market_returns=market_returns,
        current_weights={},
        anomalies=anomalies
    )
    
    print(f"\nRecommendations:")
    for rec in recommendations[:5]:
        print(f"{rec['symbol']}: {rec['action']} ({rec['confidence']:.0f}%)")
        print(f"  {rec['explanation']}\n")
    
    # Test simulator
    simulator = PortfolioSimulator(initial_capital=10000.0)
    
    # Simulate some trades
    prices = {sym: 100.0 for sym in symbols}
    for symbol, weight in opt_result.weights.items():
        if weight > 0:
            quantity = int((10000.0 * weight) / prices[symbol])
            simulator.buy(symbol, quantity, prices[symbol], date.today())
    
    simulator.record_state(date.today(), prices)
    
    metrics = simulator.calculate_metrics()
    print(f"\nSimulation Metrics:")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Initial Capital: {metrics.initial_capital:.2f} TND")
    print(f"Final Value: {metrics.final_value:.2f} TND")
    
    print("\n✅ Full pipeline test completed successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_full_pipeline())
