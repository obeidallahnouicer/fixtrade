"""
Portfolio Performance Metrics Calculator.

Implements financial metrics for portfolio evaluation:
- ROI (Return on Investment)
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (worst peak-to-trough decline)
- Volatility (standard deviation of returns)
- Win Rate (% of profitable trades)
- Average Gain/Loss per trade

These metrics help evaluate portfolio performance and compare
different trading strategies.
"""

import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.ai.config import ai_settings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Complete portfolio performance metrics."""
    
    # Basic metrics
    total_value: float
    total_return: float  # Absolute return
    roi: float  # Return on Investment (%)
    
    # Risk metrics
    volatility: float  # Annualized volatility
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Maximum peak-to-trough decline (%)
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # %
    avg_gain: float
    avg_loss: float
    profit_factor: float  # Total gains / Total losses
    
    # Time-based
    days_active: int
    annualized_return: float


class MetricsCalculator:
    """Calculate portfolio performance metrics."""
    
    def __init__(
        self,
        risk_free_rate: float = ai_settings.risk_free_rate,
        trading_days_per_year: int = ai_settings.trading_days_per_year
    ):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
            trading_days_per_year: Number of trading days (default: 250)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def calculate_roi(
        self,
        initial_capital: float,
        current_value: float
    ) -> float:
        """
        Calculate Return on Investment.
        
        ROI = (Current Value - Initial Capital) / Initial Capital * 100
        """
        if initial_capital == 0:
            return 0.0
        return ((current_value - initial_capital) / initial_capital) * 100
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return).
        
        Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
        Results are annualized.
        
        Args:
            returns: List of periodic returns (e.g., daily returns)
            periods_per_year: Periods in a year (e.g., 250 for daily)
        
        Returns:
            Annualized Sharpe Ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        if periods_per_year is None:
            periods_per_year = self.trading_days_per_year
        
        returns_array = np.array(returns)
        
        # Calculate mean and std of returns
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Daily risk-free rate
        daily_rf_rate = self.risk_free_rate / periods_per_year
        
        # Sharpe ratio (annualized)
        sharpe = ((mean_return - daily_rf_rate) / std_return) * math.sqrt(periods_per_year)
        
        return round(sharpe, 4)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate Maximum Drawdown.
        
        Max Drawdown = (Trough Value - Peak Value) / Peak Value * 100
        
        This represents the worst peak-to-trough decline.
        
        Args:
            portfolio_values: Time series of portfolio values
        
        Returns:
            Maximum drawdown as percentage (negative value)
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        # Return maximum drawdown (most negative)
        max_dd = np.min(drawdowns) * 100
        
        return round(max_dd, 2)
    
    def calculate_volatility(
        self,
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            returns: List of periodic returns
            annualize: Whether to annualize the volatility
        
        Returns:
            Volatility (annualized if requested)
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        std_dev = np.std(returns, ddof=1)
        
        if annualize:
            std_dev *= math.sqrt(self.trading_days_per_year)
        
        return round(std_dev * 100, 2)  # Return as percentage
    
    def calculate_trade_statistics(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate trade-level statistics.
        
        Args:
            trades: List of trade dictionaries with 'profit_loss' key
        
        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_gain": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        profits = [t["profit_loss"] for t in trades if t["profit_loss"] > 0]
        losses = [t["profit_loss"] for t in trades if t["profit_loss"] < 0]
        
        total_trades = len(trades)
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_gain = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_gains = sum(profits)
        total_losses = abs(sum(losses))
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "avg_gain": round(avg_gain, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2)
        }
    
    def calculate_annualized_return(
        self,
        total_return: float,
        days_active: int
    ) -> float:
        """
        Calculate annualized return.
        
        Annualized Return = (1 + Total Return) ^ (365 / Days) - 1
        """
        if days_active == 0:
            return 0.0
        
        annual_return = (
            math.pow(1 + total_return, 365 / days_active) - 1
        ) * 100
        
        return round(annual_return, 2)
    
    def calculate_all_metrics(
        self,
        initial_capital: float,
        current_value: float,
        portfolio_values: List[float],
        returns: List[float],
        trades: List[Dict[str, Any]],
        start_date: date,
        end_date: Optional[date] = None
    ) -> PortfolioMetrics:
        """
        Calculate all portfolio metrics at once.
        
        Args:
            initial_capital: Starting capital
            current_value: Current portfolio value
            portfolio_values: Time series of portfolio values
            returns: List of periodic returns
            trades: List of completed trades
            start_date: Portfolio start date
            end_date: Current date (default: today)
        
        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        if end_date is None:
            end_date = date.today()
        
        days_active = (end_date - start_date).days
        
        # Basic metrics
        total_return = (current_value - initial_capital) / initial_capital
        roi = self.calculate_roi(initial_capital, current_value)
        
        # Risk metrics
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Trade statistics
        trade_stats = self.calculate_trade_statistics(trades)
        
        # Annualized return
        annualized_return = self.calculate_annualized_return(total_return, days_active)
        
        metrics = PortfolioMetrics(
            total_value=round(current_value, 2),
            total_return=round(total_return * 100, 2),
            roi=roi,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            avg_gain=trade_stats["avg_gain"],
            avg_loss=trade_stats["avg_loss"],
            profit_factor=trade_stats["profit_factor"],
            days_active=days_active,
            annualized_return=annualized_return
        )
        
        logger.info(f"Calculated metrics: ROI={roi:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return metrics
