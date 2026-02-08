"""
Basic tests for AI module components.

Run with: pytest app/ai/test_basic.py -v
"""

import pytest
from datetime import datetime, date
from uuid import uuid4

from app.ai.profile import RiskProfile, UserProfileManager, ProfileCharacteristics
from app.ai.portfolio import PortfolioManager, Position, Trade
from app.ai.metrics import MetricsCalculator
from app.ai.rules import RuleBasedEngine, Signal, SignalStrength, MarketSignals


class TestUserProfile:
    """Test user profile management."""
    
    def test_profile_characteristics(self):
        """Test getting profile characteristics."""
        manager = UserProfileManager()
        
        conservative = manager.get_characteristics(RiskProfile.CONSERVATIVE)
        assert conservative.max_position_size == 0.10
        assert conservative.max_equity_allocation == 0.40
        
        aggressive = manager.get_characteristics(RiskProfile.AGGRESSIVE)
        assert aggressive.max_position_size == 0.25
        assert aggressive.max_equity_allocation == 0.90
    
    def test_profile_recommendation(self):
        """Test profile recommendation from questionnaire."""
        manager = UserProfileManager()
        
        # Conservative profile
        questionnaire = {
            "age": 60,
            "investment_horizon": 2,
            "income_stability": "low",
            "investment_experience": "beginner",
            "loss_tolerance": 1,
            "financial_goals": "preservation"
        }
        
        recommended = manager.recommend_profile(questionnaire)
        assert recommended == RiskProfile.CONSERVATIVE
        
        # Aggressive profile
        questionnaire = {
            "age": 25,
            "investment_horizon": 15,
            "income_stability": "high",
            "investment_experience": "expert",
            "loss_tolerance": 5,
            "financial_goals": "aggressive_growth"
        }
        
        recommended = manager.recommend_profile(questionnaire)
        assert recommended == RiskProfile.AGGRESSIVE


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_initialization(self):
        """Test portfolio creation."""
        portfolio = PortfolioManager(
            risk_profile=RiskProfile.MODERATE,
            initial_capital=10000.0
        )
        
        assert portfolio.initial_capital == 10000.0
        assert portfolio.cash_balance == 10000.0
        assert portfolio.total_value == 10000.0
        assert len(portfolio.positions) == 0
    
    def test_buy_operation(self):
        """Test buying stocks."""
        portfolio = PortfolioManager(
            risk_profile=RiskProfile.MODERATE,
            initial_capital=10000.0
        )
        
        success, message = portfolio.buy("TEST", 100, 10.0)
        
        assert success is True
        assert portfolio.cash_balance == 9000.0
        assert "TEST" in portfolio.positions
        assert portfolio.positions["TEST"].quantity == 100
        assert len(portfolio.trade_history) == 1
    
    def test_sell_operation(self):
        """Test selling stocks."""
        portfolio = PortfolioManager(
            risk_profile=RiskProfile.MODERATE,
            initial_capital=10000.0
        )
        
        # Buy first
        portfolio.buy("TEST", 100, 10.0)
        
        # Sell at profit
        success, message, pnl = portfolio.sell("TEST", 100, 12.0)
        
        assert success is True
        assert pnl == 200.0  # (12 - 10) * 100
        assert portfolio.cash_balance == 10200.0
        assert "TEST" not in portfolio.positions
    
    def test_position_size_limit(self):
        """Test position size limits."""
        portfolio = PortfolioManager(
            risk_profile=RiskProfile.MODERATE,  # Max 15% per position
            initial_capital=10000.0
        )
        
        # Try to buy 20% of portfolio (should fail)
        success, message = portfolio.buy("TEST", 200, 10.0)
        # 200 * 10 = 2000 / 10000 = 20% > 15%
        
        assert success is False
        assert "Position trop importante" in message
    
    def test_stop_loss_detection(self):
        """Test stop-loss detection."""
        portfolio = PortfolioManager(
            risk_profile=RiskProfile.MODERATE,  # 8% stop-loss
            initial_capital=10000.0
        )
        
        # Buy at 10.0
        portfolio.buy("TEST", 100, 10.0)
        
        # Price drops to 9.0 (-10%)
        portfolio.update_prices({"TEST": 9.0})
        
        triggered = portfolio.check_stop_losses()
        
        assert "TEST" in triggered


class TestMetrics:
    """Test metrics calculation."""
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        calc = MetricsCalculator()
        
        roi = calc.calculate_roi(
            initial_capital=10000.0,
            current_value=12000.0
        )
        
        assert roi == 20.0  # 20% return
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator(risk_free_rate=0.05)
        
        # Example daily returns
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.03]
        
        sharpe = calc.calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should give positive Sharpe
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        
        # Portfolio values: peak at 12000, trough at 9000
        values = [10000, 11000, 12000, 11000, 9000, 10000]
        
        max_dd = calc.calculate_max_drawdown(values)
        
        # (9000 - 12000) / 12000 = -25%
        assert max_dd == -25.0
    
    def test_trade_statistics(self):
        """Test trade statistics."""
        calc = MetricsCalculator()
        
        trades = [
            {"profit_loss": 100},
            {"profit_loss": 50},
            {"profit_loss": -30},
            {"profit_loss": 200},
            {"profit_loss": -50}
        ]
        
        stats = calc.calculate_trade_statistics(trades)
        
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == 60.0


class TestRules:
    """Test rule-based engine."""
    
    def test_strong_buy_signal(self):
        """Test strong buy signal generation."""
        engine = RuleBasedEngine(risk_profile=RiskProfile.MODERATE)
        
        signals = MarketSignals(
            symbol="TEST",
            predicted_return=6.0,  # +6%
            confidence_score=0.85,
            sentiment_score=0.70,  # Very positive
            liquidity_tier="high"
        )
        
        signal, strength, reasons = engine.evaluate(signals)
        
        assert signal in [Signal.BUY, Signal.STRONG_BUY]
        assert len(reasons) > 0
    
    def test_sell_signal_with_anomaly(self):
        """Test sell signal when anomaly detected."""
        engine = RuleBasedEngine(risk_profile=RiskProfile.CONSERVATIVE)
        
        signals = MarketSignals(
            symbol="TEST",
            predicted_return=-3.0,
            confidence_score=0.70,
            has_anomaly=True,
            anomaly_severity=0.80,
            anomaly_type="volume_spike"
        )
        
        signal, strength, reasons = engine.evaluate(signals)
        
        assert signal in [Signal.SELL, Signal.STRONG_SELL, Signal.HOLD]
        assert any("Anomalie" in reason for reason in reasons)
    
    def test_hold_signal(self):
        """Test hold signal for neutral conditions."""
        engine = RuleBasedEngine(risk_profile=RiskProfile.MODERATE)
        
        signals = MarketSignals(
            symbol="TEST",
            predicted_return=0.5,  # Small gain
            confidence_score=0.60,  # Medium confidence
            sentiment_score=0.0    # Neutral
        )
        
        signal, strength, reasons = engine.evaluate(signals)
        
        # Could be BUY or HOLD depending on thresholds
        assert signal in [Signal.BUY, Signal.HOLD]


class TestPosition:
    """Test Position data class."""
    
    def test_position_calculations(self):
        """Test position P&L calculations."""
        position = Position(
            symbol="TEST",
            quantity=100,
            purchase_price=10.0,
            purchased_at=date.today(),
            current_price=12.0
        )
        
        assert position.cost_basis == 1000.0
        assert position.current_value == 1200.0
        assert position.unrealized_pnl == 200.0
        assert position.unrealized_pnl_pct == 20.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
