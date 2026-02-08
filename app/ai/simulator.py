"""
Portfolio Simulation Engine.

Simulates portfolio performance over time with:
- Virtual trading
- Daily rebalancing
- Performance tracking
- Risk metrics
- Transaction costs (optional)

All calculations are deterministic.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from app.ai.profile import RiskProfile

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Single portfolio transaction."""
    date: date
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total transaction cost."""
        return (self.quantity * self.price) + self.commission


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""
    date: date
    cash: float
    positions: Dict[str, int]
    prices: Dict[str, float]
    total_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0


@dataclass
class SimulationMetrics:
    """Complete simulation metrics."""
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_gain: float
    avg_loss: float
    total_commissions: float
    
    value_history: List[float] = field(default_factory=list)
    return_history: List[float] = field(default_factory=list)
    dates: List[date] = field(default_factory=list)


class PortfolioSimulator:
    """
    Simulates portfolio performance over time.
    
    Supports backtesting of trading strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 250
    ):
        """
        Initialize simulator.
        
        Args:
            initial_capital: Starting capital in TND
            commission_rate: Commission as fraction of trade value
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days for annualization
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.transactions: List[Transaction] = []
        self.state_history: List[PortfolioState] = []
    
    def reset(self):
        """Reset simulator to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.transactions = []
        self.state_history = []
    
    def get_position_value(
        self,
        symbol: str,
        price: float
    ) -> float:
        """Calculate value of a position."""
        quantity = self.positions.get(symbol, 0)
        return quantity * price
    
    def get_total_value(
        self,
        prices: Dict[str, float]
    ) -> float:
        """Calculate total portfolio value."""
        equity_value = sum(
            self.get_position_value(symbol, prices[symbol])
            for symbol in self.positions
            if symbol in prices
        )
        return self.cash + equity_value
    
    def can_buy(
        self,
        symbol: str,
        quantity: int,
        price: float
    ) -> bool:
        """Check if buy order is possible."""
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        return total_cost <= self.cash
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        trade_date: date
    ) -> bool:
        """
        Execute buy order.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares
            price: Price per share
            trade_date: Transaction date
        
        Returns:
            True if successful
        """
        if not self.can_buy(symbol, quantity, price):
            logger.warning(
                f"Cannot buy {quantity} {symbol} @ {price:.2f}: "
                f"insufficient cash ({self.cash:.2f})"
            )
            return False
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        transaction = Transaction(
            date=trade_date,
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            commission=commission
        )
        self.transactions.append(transaction)
        
        logger.info(
            f"BUY {quantity} {symbol} @ {price:.2f} "
            f"(commission: {commission:.2f}, cash: {self.cash:.2f})"
        )
        
        return True
    
    def can_sell(
        self,
        symbol: str,
        quantity: int
    ) -> bool:
        """Check if sell order is possible."""
        return self.positions.get(symbol, 0) >= quantity
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        trade_date: date
    ) -> bool:
        """
        Execute sell order.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares
            price: Price per share
            trade_date: Transaction date
        
        Returns:
            True if successful
        """
        if not self.can_sell(symbol, quantity):
            logger.warning(
                f"Cannot sell {quantity} {symbol}: "
                f"insufficient position ({self.positions.get(symbol, 0)})"
            )
            return False
        
        proceeds = quantity * price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        self.cash += net_proceeds
        self.positions[symbol] -= quantity
        
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        transaction = Transaction(
            date=trade_date,
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            commission=commission
        )
        self.transactions.append(transaction)
        
        logger.info(
            f"SELL {quantity} {symbol} @ {price:.2f} "
            f"(commission: {commission:.2f}, cash: {self.cash:.2f})"
        )
        
        return True
    
    def rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        trade_date: date,
        threshold: float = 0.01
    ) -> int:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dict of symbol -> target weight
            prices: Current prices
            trade_date: Transaction date
            threshold: Minimum weight delta to trade
        
        Returns:
            Number of trades executed
        """
        total_value = self.get_total_value(prices)
        trades_executed = 0
        
        current_weights = {}
        for symbol in target_weights.keys():
            if symbol in prices:
                position_value = self.get_position_value(symbol, prices[symbol])
                current_weights[symbol] = position_value / total_value if total_value > 0 else 0.0
            else:
                current_weights[symbol] = 0.0
        
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue
            
            current_weight = current_weights.get(symbol, 0.0)
            weight_delta = target_weight - current_weight
            
            if abs(weight_delta) < threshold:
                continue
            
            target_value = total_value * target_weight
            current_quantity = self.positions.get(symbol, 0)
            target_quantity = int(target_value / prices[symbol])
            quantity_delta = target_quantity - current_quantity
            
            if quantity_delta > 0:
                success = self.buy(symbol, quantity_delta, prices[symbol], trade_date)
                if success:
                    trades_executed += 1
            
            elif quantity_delta < 0:
                success = self.sell(symbol, abs(quantity_delta), prices[symbol], trade_date)
                if success:
                    trades_executed += 1
        
        return trades_executed
    
    def record_state(
        self,
        trade_date: date,
        prices: Dict[str, float]
    ):
        """Record current portfolio state."""
        total_value = self.get_total_value(prices)
        
        daily_return = 0.0
        cumulative_return = 0.0
        
        if len(self.state_history) > 0:
            prev_value = self.state_history[-1].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value
        
        cumulative_return = (total_value - self.initial_capital) / self.initial_capital
        
        state = PortfolioState(
            date=trade_date,
            cash=self.cash,
            positions=self.positions.copy(),
            prices=prices.copy(),
            total_value=total_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return
        )
        
        self.state_history.append(state)
    
    def calculate_metrics(self) -> SimulationMetrics:
        """Calculate complete simulation metrics."""
        if not self.state_history:
            logger.warning("No state history to calculate metrics")
            return SimulationMetrics(
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0.0,
                total_return_pct=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_gain=0.0,
                avg_loss=0.0,
                total_commissions=0.0
            )
        
        final_value = self.state_history[-1].total_value
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        days = len(self.state_history)
        years = days / self.trading_days_per_year
        annualized_return = (
            ((final_value / self.initial_capital) ** (1 / years)) - 1
            if years > 0 else 0.0
        )
        
        returns = [state.daily_return for state in self.state_history]
        volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)
        
        sharpe_ratio = (
            (annualized_return - self.risk_free_rate) / volatility
            if volatility > 0 else 0.0
        )
        
        downside_returns = [r for r in returns if r < 0]
        downside_std = (
            np.std(downside_returns) * np.sqrt(self.trading_days_per_year)
            if downside_returns else 0.0
        )
        sortino_ratio = (
            (annualized_return - self.risk_free_rate) / downside_std
            if downside_std > 0 else 0.0
        )
        
        values = [state.total_value for state in self.state_history]
        running_max = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        buy_transactions = [t for t in self.transactions if t.action == "BUY"]
        sell_transactions = [t for t in self.transactions if t.action == "SELL"]
        
        trade_pnl = []
        for sell in sell_transactions:
            matching_buys = [
                b for b in buy_transactions
                if b.symbol == sell.symbol and b.date <= sell.date
            ]
            if matching_buys:
                avg_buy_price = np.mean([b.price for b in matching_buys])
                pnl = (sell.price - avg_buy_price) * sell.quantity - sell.commission
                trade_pnl.append(pnl)
        
        winning_trades = len([pnl for pnl in trade_pnl if pnl > 0])
        losing_trades = len([pnl for pnl in trade_pnl if pnl < 0])
        total_trades = len(trade_pnl)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        gains = [pnl for pnl in trade_pnl if pnl > 0]
        losses = [abs(pnl) for pnl in trade_pnl if pnl < 0]
        
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_gains = sum(gains)
        total_losses = sum(losses)
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 0.0
        
        total_commissions = sum(t.commission for t in self.transactions)
        
        return SimulationMetrics(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct * 100,
            annualized_return=annualized_return * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown * 100,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            total_commissions=total_commissions,
            value_history=values,
            return_history=returns,
            dates=[state.date for state in self.state_history]
        )
