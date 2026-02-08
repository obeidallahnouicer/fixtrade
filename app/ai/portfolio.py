"""
Portfolio Management and Simulation Engine.

Manages virtual portfolios with:
- Position tracking (buy/sell operations)
- Cash balance management
- Performance tracking
- Risk management (stop-loss, position limits)
- Historical value tracking for metrics calculation

Integrates with database to persist portfolio state.
"""

import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.config import ai_settings
from app.ai.profile import RiskProfile, UserProfileManager
from app.ai.metrics import MetricsCalculator, PortfolioMetrics

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a stock position in the portfolio."""
    symbol: str
    quantity: int
    purchase_price: float
    purchased_at: date
    current_price: Optional[float] = None
    
    @property
    def cost_basis(self) -> float:
        """Total cost of the position."""
        return self.quantity * self.purchase_price
    
    @property
    def current_value(self) -> float:
        """Current market value of the position."""
        if self.current_price is None:
            return self.cost_basis
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.current_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    executed_at: datetime
    profit_loss: Optional[float] = None  # For SELL trades


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    portfolio_id: UUID
    timestamp: datetime
    total_value: float
    cash_balance: float
    equity_value: float
    positions: List[Position] = field(default_factory=list)


class PortfolioManager:
    """
    Manages portfolio operations and simulation.
    
    Handles:
    - Position management (buy/sell)
    - Cash flow tracking
    - Risk management
    - Performance calculation
    """
    
    def __init__(
        self,
        portfolio_id: Optional[UUID] = None,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        initial_capital: float = ai_settings.default_initial_capital
    ):
        """
        Initialize portfolio manager.
        
        Args:
            portfolio_id: Existing portfolio ID (None for new)
            risk_profile: Investment risk profile
            initial_capital: Starting capital in TND
        """
        self.portfolio_id = portfolio_id or uuid4()
        self.risk_profile = risk_profile
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.value_history: List[Tuple[datetime, float]] = []
        
        self.profile_manager = UserProfileManager()
        self.metrics_calculator = MetricsCalculator()
        
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        logger.info(
            f"Portfolio {self.portfolio_id} initialized: "
            f"{risk_profile}, {initial_capital} TND"
        )
    
    @property
    def equity_value(self) -> float:
        """Total value of all positions."""
        return sum(pos.current_value for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + equity)."""
        return self.cash_balance + self.equity_value
    
    @property
    def equity_allocation(self) -> float:
        """Current equity allocation as percentage."""
        if self.total_value == 0:
            return 0.0
        return self.equity_value / self.total_value
    
    def can_buy(
        self,
        symbol: str,
        quantity: int,
        price: float
    ) -> Tuple[bool, str]:
        """
        Check if a buy order is allowed.
        
        Validates:
        - Sufficient cash
        - Position size limits
        - Equity allocation limits
        
        Returns:
            (is_allowed, reason)
        """
        cost = quantity * price
        
        # Check cash availability
        if cost > self.cash_balance:
            return False, f"Fonds insuffisants: {cost:.2f} TND requis, {self.cash_balance:.2f} TND disponibles"
        
        # Check position size limit
        position_size = cost / self.total_value
        chars = self.profile_manager.get_characteristics(self.risk_profile)
        
        if position_size > chars.max_position_size:
            return False, (
                f"Position trop importante: {position_size:.1%} "
                f"(max: {chars.max_position_size:.1%})"
            )
        
        # Check equity allocation limit
        new_equity_value = self.equity_value + cost
        new_total_value = self.total_value  # Cash decreases, equity increases
        new_allocation = new_equity_value / new_total_value
        
        if new_allocation > chars.max_equity_allocation:
            return False, (
                f"Allocation actions trop élevée: {new_allocation:.1%} "
                f"(max: {chars.max_equity_allocation:.1%})"
            )
        
        return True, "Achat autorisé"
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        executed_at: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Execute a buy order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            executed_at: Execution timestamp (default: now)
        
        Returns:
            (success, message)
        """
        if executed_at is None:
            executed_at = datetime.now()
        
        can_buy, reason = self.can_buy(symbol, quantity, price)
        if not can_buy:
            logger.warning(f"Buy rejected for {symbol}: {reason}")
            return False, reason
        
        cost = quantity * price
        
        # Update cash
        self.cash_balance -= cost
        
        # Update or create position
        if symbol in self.positions:
            # Average cost for existing position
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            avg_price = (
                (existing.quantity * existing.purchase_price + cost) / total_quantity
            )
            existing.quantity = total_quantity
            existing.purchase_price = avg_price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                purchase_price=price,
                purchased_at=date.today(),
                current_price=price
            )
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            executed_at=executed_at
        )
        self.trade_history.append(trade)
        
        self.updated_at = executed_at
        
        logger.info(
            f"BUY executed: {quantity} {symbol} @ {price:.3f} TND "
            f"(cost: {cost:.2f} TND)"
        )
        
        return True, f"Achat réussi: {quantity} {symbol} @ {price:.3f} TND"
    
    def can_sell(
        self,
        symbol: str,
        quantity: int
    ) -> Tuple[bool, str]:
        """
        Check if a sell order is allowed.
        
        Returns:
            (is_allowed, reason)
        """
        if symbol not in self.positions:
            return False, f"Aucune position pour {symbol}"
        
        position = self.positions[symbol]
        if quantity > position.quantity:
            return False, (
                f"Quantité insuffisante: {quantity} demandé, "
                f"{position.quantity} disponible"
            )
        
        return True, "Vente autorisée"
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        executed_at: Optional[datetime] = None
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Execute a sell order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            executed_at: Execution timestamp (default: now)
        
        Returns:
            (success, message, profit_loss)
        """
        if executed_at is None:
            executed_at = datetime.now()
        
        can_sell, reason = self.can_sell(symbol, quantity)
        if not can_sell:
            logger.warning(f"Sell rejected for {symbol}: {reason}")
            return False, reason, None
        
        position = self.positions[symbol]
        proceeds = quantity * price
        cost_basis = quantity * position.purchase_price
        profit_loss = proceeds - cost_basis
        
        # Update cash
        self.cash_balance += proceeds
        
        # Update position
        position.quantity -= quantity
        if position.quantity == 0:
            del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            executed_at=executed_at,
            profit_loss=profit_loss
        )
        self.trade_history.append(trade)
        
        self.updated_at = executed_at
        
        logger.info(
            f"SELL executed: {quantity} {symbol} @ {price:.3f} TND "
            f"(P&L: {profit_loss:.2f} TND)"
        )
        
        return True, f"Vente réussie: {quantity} {symbol} @ {price:.3f} TND", profit_loss
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for positions.
        
        Args:
            prices: Dict mapping symbol -> current_price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
        
        # Record value snapshot
        self.value_history.append((datetime.now(), self.total_value))
    
    def check_stop_losses(self) -> List[str]:
        """
        Check all positions for stop-loss triggers.
        
        Returns:
            List of symbols that hit stop-loss
        """
        chars = self.profile_manager.get_characteristics(self.risk_profile)
        stop_loss_threshold = chars.stop_loss_threshold
        
        triggered = []
        
        for symbol, position in self.positions.items():
            if position.current_price is None:
                continue
            
            loss_pct = position.unrealized_pnl_pct
            
            if loss_pct <= -stop_loss_threshold * 100:
                triggered.append(symbol)
                logger.warning(
                    f"Stop-loss triggered for {symbol}: "
                    f"{loss_pct:.2f}% loss (threshold: {stop_loss_threshold*100:.0f}%)"
                )
        
        return triggered
    
    def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        return PortfolioSnapshot(
            portfolio_id=self.portfolio_id,
            timestamp=datetime.now(),
            total_value=self.total_value,
            cash_balance=self.cash_balance,
            equity_value=self.equity_value,
            positions=list(self.positions.values())
        )
    
    def calculate_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        
        # Extract returns from value history
        values = [v for _, v in self.value_history]
        if len(values) < 2:
            values = [self.initial_capital, self.total_value]
        
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        # Extract completed trades (only SELL trades have P&L)
        completed_trades = [
            {"profit_loss": t.profit_loss}
            for t in self.trade_history
            if t.action == "SELL" and t.profit_loss is not None
        ]
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            initial_capital=self.initial_capital,
            current_value=self.total_value,
            portfolio_values=values,
            returns=returns,
            trades=completed_trades,
            start_date=self.created_at.date(),
            end_date=date.today()
        )
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio state to dictionary."""
        return {
            "portfolio_id": str(self.portfolio_id),
            "risk_profile": self.risk_profile.value,
            "initial_capital": self.initial_capital,
            "cash_balance": self.cash_balance,
            "equity_value": self.equity_value,
            "total_value": self.total_value,
            "equity_allocation": self.equity_allocation,
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
                for pos in self.positions.values()
            ],
            "trade_count": len(self.trade_history),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
