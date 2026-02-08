# AI Module - Specific Code Improvements

This document contains **ready-to-implement code snippets** for the most critical improvements to the AI module.

---

## 1. Database Persistence Layer

### Step 1.1: Create Database Models

**File**: `app/domain/trading/portfolio_models.py` (NEW)

```python
"""
Portfolio domain models for database persistence.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Date, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4

from app.core.db import Base


class Portfolio(Base):
    """Portfolio entity."""
    
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=True, index=True)
    risk_profile = Column(String(50), nullable=False)
    initial_capital = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    positions = relationship("PortfolioPosition", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("PortfolioTrade", back_populates="portfolio", cascade="all, delete-orphan")
    snapshots = relationship("PortfolioSnapshot", back_populates="portfolio", cascade="all, delete-orphan")


class PortfolioPosition(Base):
    """Portfolio position entity."""
    
    __tablename__ = "portfolio_positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchased_at = Column(Date, nullable=False)
    current_price = Column(Float, nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")


class PortfolioTrade(Base):
    """Portfolio trade entity."""
    
    __tablename__ = "portfolio_trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # BUY or SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    executed_at = Column(DateTime, nullable=False)
    profit_loss = Column(Float, nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="trades")


class PortfolioSnapshot(Base):
    """Portfolio value snapshot for performance tracking."""
    
    __tablename__ = "portfolio_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    equity_value = Column(Float, nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="snapshots")
```

### Step 1.2: Create Migration Script

**File**: `db/002_portfolio_tables.sql` (NEW)

```sql
-- Portfolio tables for AI module
-- Migration: 002
-- Created: 2026-02-08

BEGIN;

-- Portfolios table
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    risk_profile VARCHAR(50) NOT NULL CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    initial_capital FLOAT NOT NULL CHECK (initial_capital > 0),
    cash_balance FLOAT NOT NULL CHECK (cash_balance >= 0),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    purchase_price FLOAT NOT NULL CHECK (purchase_price > 0),
    purchased_at DATE NOT NULL,
    current_price FLOAT
);

CREATE INDEX idx_positions_portfolio_id ON portfolio_positions(portfolio_id);
CREATE INDEX idx_positions_symbol ON portfolio_positions(symbol);

-- Portfolio trades table
CREATE TABLE IF NOT EXISTS portfolio_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    price FLOAT NOT NULL CHECK (price > 0),
    executed_at TIMESTAMP NOT NULL,
    profit_loss FLOAT
);

CREATE INDEX idx_trades_portfolio_id ON portfolio_trades(portfolio_id);
CREATE INDEX idx_trades_symbol ON portfolio_trades(symbol);
CREATE INDEX idx_trades_executed_at ON portfolio_trades(executed_at);

-- Portfolio snapshots table (for performance tracking)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    total_value FLOAT NOT NULL,
    cash_balance FLOAT NOT NULL,
    equity_value FLOAT NOT NULL
);

CREATE INDEX idx_snapshots_portfolio_id ON portfolio_snapshots(portfolio_id);
CREATE INDEX idx_snapshots_timestamp ON portfolio_snapshots(timestamp);

-- Update trigger for portfolios.updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMIT;
```

### Step 1.3: Create Repository

**File**: `app/infrastructure/trading/portfolio_repository.py` (NEW)

```python
"""
Repository for portfolio persistence operations.
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.trading.portfolio_models import (
    Portfolio as PortfolioDB,
    PortfolioPosition as PositionDB,
    PortfolioTrade as TradeDB,
    PortfolioSnapshot as SnapshotDB
)
from app.ai.portfolio import PortfolioManager, Position, Trade
from app.ai.profile import RiskProfile

logger = logging.getLogger(__name__)


class PortfolioRepository:
    """Repository for portfolio operations."""
    
    async def save_portfolio(
        self,
        session: AsyncSession,
        portfolio: PortfolioManager
    ) -> None:
        """
        Persist portfolio to database.
        
        Upserts portfolio and all positions/trades.
        """
        try:
            # Check if portfolio exists
            result = await session.execute(
                select(PortfolioDB).where(PortfolioDB.id == portfolio.portfolio_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing
                await session.execute(
                    update(PortfolioDB)
                    .where(PortfolioDB.id == portfolio.portfolio_id)
                    .values(
                        cash_balance=portfolio.cash_balance,
                        updated_at=datetime.now()
                    )
                )
            else:
                # Create new
                db_portfolio = PortfolioDB(
                    id=portfolio.portfolio_id,
                    risk_profile=portfolio.risk_profile.value,
                    initial_capital=portfolio.initial_capital,
                    cash_balance=portfolio.cash_balance,
                    created_at=portfolio.created_at,
                    updated_at=portfolio.updated_at
                )
                session.add(db_portfolio)
            
            # Delete existing positions (we'll re-create them)
            await session.execute(
                delete(PositionDB).where(PositionDB.portfolio_id == portfolio.portfolio_id)
            )
            
            # Save current positions
            for pos in portfolio.positions.values():
                db_position = PositionDB(
                    portfolio_id=portfolio.portfolio_id,
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    purchase_price=pos.purchase_price,
                    purchased_at=pos.purchased_at,
                    current_price=pos.current_price
                )
                session.add(db_position)
            
            # Save new trades (only ones not yet persisted)
            existing_trade_count = await session.scalar(
                select(TradeDB)
                .where(TradeDB.portfolio_id == portfolio.portfolio_id)
                .count()
            )
            
            new_trades = portfolio.trade_history[existing_trade_count:]
            for trade in new_trades:
                db_trade = TradeDB(
                    portfolio_id=portfolio.portfolio_id,
                    symbol=trade.symbol,
                    action=trade.action,
                    quantity=trade.quantity,
                    price=trade.price,
                    executed_at=trade.executed_at,
                    profit_loss=trade.profit_loss
                )
                session.add(db_trade)
            
            # Save snapshot
            db_snapshot = SnapshotDB(
                portfolio_id=portfolio.portfolio_id,
                timestamp=datetime.now(),
                total_value=portfolio.total_value,
                cash_balance=portfolio.cash_balance,
                equity_value=portfolio.equity_value
            )
            session.add(db_snapshot)
            
            await session.commit()
            logger.info(f"Portfolio {portfolio.portfolio_id} saved to database")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error saving portfolio: {e}")
            raise
    
    async def load_portfolio(
        self,
        session: AsyncSession,
        portfolio_id: UUID
    ) -> Optional[PortfolioManager]:
        """
        Load portfolio from database.
        
        Returns None if portfolio not found.
        """
        try:
            # Load portfolio with relationships
            result = await session.execute(
                select(PortfolioDB)
                .options(
                    selectinload(PortfolioDB.positions),
                    selectinload(PortfolioDB.trades)
                )
                .where(PortfolioDB.id == portfolio_id)
            )
            db_portfolio = result.scalar_one_or_none()
            
            if not db_portfolio:
                return None
            
            # Create PortfolioManager instance
            portfolio = PortfolioManager(
                portfolio_id=db_portfolio.id,
                risk_profile=RiskProfile(db_portfolio.risk_profile),
                initial_capital=db_portfolio.initial_capital
            )
            
            # Restore cash balance
            portfolio.cash_balance = db_portfolio.cash_balance
            
            # Restore positions
            portfolio.positions = {}
            for db_pos in db_portfolio.positions:
                position = Position(
                    symbol=db_pos.symbol,
                    quantity=db_pos.quantity,
                    purchase_price=db_pos.purchase_price,
                    purchased_at=db_pos.purchased_at,
                    current_price=db_pos.current_price
                )
                portfolio.positions[db_pos.symbol] = position
            
            # Restore trade history
            portfolio.trade_history = []
            for db_trade in sorted(db_portfolio.trades, key=lambda t: t.executed_at):
                trade = Trade(
                    symbol=db_trade.symbol,
                    action=db_trade.action,
                    quantity=db_trade.quantity,
                    price=db_trade.price,
                    executed_at=db_trade.executed_at,
                    profit_loss=db_trade.profit_loss
                )
                portfolio.trade_history.append(trade)
            
            # Restore timestamps
            portfolio.created_at = db_portfolio.created_at
            portfolio.updated_at = db_portfolio.updated_at
            
            logger.info(f"Portfolio {portfolio_id} loaded from database")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            raise
    
    async def delete_portfolio(
        self,
        session: AsyncSession,
        portfolio_id: UUID
    ) -> bool:
        """Delete portfolio (cascade deletes positions/trades)."""
        try:
            result = await session.execute(
                delete(PortfolioDB).where(PortfolioDB.id == portfolio_id)
            )
            await session.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Portfolio {portfolio_id} deleted")
            
            return deleted
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error deleting portfolio: {e}")
            raise
    
    async def get_portfolio_snapshots(
        self,
        session: AsyncSession,
        portfolio_id: UUID,
        limit: int = 100
    ) -> List[dict]:
        """Get historical snapshots for performance chart."""
        try:
            result = await session.execute(
                select(SnapshotDB)
                .where(SnapshotDB.portfolio_id == portfolio_id)
                .order_by(SnapshotDB.timestamp.desc())
                .limit(limit)
            )
            snapshots = result.scalars().all()
            
            return [
                {
                    "timestamp": snap.timestamp.isoformat(),
                    "total_value": snap.total_value,
                    "cash_balance": snap.cash_balance,
                    "equity_value": snap.equity_value
                }
                for snap in reversed(snapshots)  # Chronological order
            ]
            
        except Exception as e:
            logger.error(f"Error loading snapshots: {e}")
            raise
```

---

## 2. Complete Data Aggregator

### Step 2.1: Implement Database Queries

**File**: `app/ai/aggregator.py` (UPDATE)

Add these implementations to replace placeholders:

```python
async def _get_current_price(
    self,
    symbol: str,
    session: AsyncSession
) -> Optional[float]:
    """Fetch latest price from stock_prices table."""
    try:
        from app.domain.trading.models import StockPrice  # Your actual model
        
        query = (
            select(StockPrice.close)
            .where(StockPrice.code == symbol)
            .order_by(desc(StockPrice.date))
            .limit(1)
        )
        
        result = await session.execute(query)
        price = result.scalar_one_or_none()
        
        return float(price) if price else None
        
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


async def _get_sentiment(
    self,
    symbol: str,
    session: AsyncSession
) -> Optional[Dict[str, Any]]:
    """Fetch latest sentiment score from sentiment_scores table."""
    try:
        # Assuming you have a SentimentScore model
        from app.domain.trading.models import SentimentScore
        
        query = (
            select(SentimentScore)
            .where(SentimentScore.symbol == symbol)
            .order_by(desc(SentimentScore.date))
            .limit(1)
        )
        
        result = await session.execute(query)
        sentiment = result.scalar_one_or_none()
        
        if sentiment:
            return {
                "score": float(sentiment.score),
                "sentiment": sentiment.label,  # positive/negative/neutral
                "date": sentiment.date
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return None


async def _get_anomaly(
    self,
    symbol: str,
    session: AsyncSession
) -> Optional[Dict[str, Any]]:
    """Fetch latest anomaly detection result."""
    try:
        # Assuming you have an Anomaly model
        from app.domain.trading.models import Anomaly
        
        # Get anomalies from last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        
        query = (
            select(Anomaly)
            .where(
                and_(
                    Anomaly.symbol == symbol,
                    Anomaly.detected_at >= yesterday
                )
            )
            .order_by(desc(Anomaly.detected_at))
            .limit(1)
        )
        
        result = await session.execute(query)
        anomaly = result.scalar_one_or_none()
        
        if anomaly:
            return {
                "severity": float(anomaly.severity),
                "type": anomaly.anomaly_type,
                "detected_at": anomaly.detected_at
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching anomaly for {symbol}: {e}")
        return None


async def _get_prediction(
    self,
    symbol: str,
    prediction_service
) -> Optional[Dict[str, Any]]:
    """Fetch price prediction from prediction service."""
    if not prediction_service:
        return None
    
    try:
        # Assuming prediction service has a predict method
        prediction = await prediction_service.predict_price(
            symbol=symbol,
            horizon=5  # 5-day forecast
        )
        
        return {
            "predicted_close": prediction.get("predicted_close"),
            "confidence_score": prediction.get("confidence"),
            "horizon_days": 5
        }
        
    except Exception as e:
        logger.error(f"Error fetching prediction for {symbol}: {e}")
        return None


async def _get_volume_prediction(
    self,
    symbol: str,
    prediction_service
) -> Optional[Dict[str, Any]]:
    """Fetch volume prediction."""
    if not prediction_service:
        return None
    
    try:
        prediction = await prediction_service.predict_volume(symbol=symbol)
        
        return {
            "predicted_volume": prediction.get("predicted_volume"),
            "confidence": prediction.get("confidence")
        }
        
    except Exception as e:
        logger.error(f"Error fetching volume prediction for {symbol}: {e}")
        return None


async def _get_liquidity_prediction(
    self,
    symbol: str,
    prediction_service
) -> Optional[Dict[str, Any]]:
    """Fetch liquidity tier prediction."""
    if not prediction_service:
        return None
    
    try:
        prediction = await prediction_service.predict_liquidity(symbol=symbol)
        
        return {
            "predicted_tier": prediction.get("tier"),  # high/medium/low
            "prob_high": prediction.get("prob_high"),
            "prob_medium": prediction.get("prob_medium"),
            "prob_low": prediction.get("prob_low")
        }
        
    except Exception as e:
        logger.error(f"Error fetching liquidity prediction for {symbol}: {e}")
        return None
```

---

## 3. Fix API Router

### Step 3.1: Add Dependency Injection

**File**: `app/ai/router.py` (UPDATE)

Replace the router implementation:

```python
from app.core.db import get_session  # Import your DB dependency
from app.ai.portfolio_repository import PortfolioRepository

# Initialize repository
portfolio_repo = PortfolioRepository()

# Dependency for prediction service
async def get_prediction_service():
    """Get prediction service instance."""
    # Import your prediction service
    from prediction.inference import PredictionService
    return PredictionService()


@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_daily_recommendations(
    session: AsyncSession = Depends(get_session),  # ✅ INJECT SESSION
    prediction_svc = Depends(get_prediction_service),  # ✅ INJECT SERVICE
    portfolio_id: str = Query(default="default"),
    top_n: int = Query(default=10, ge=1, le=50),
    symbols: Optional[str] = Query(default=None)
):
    """
    Get daily trading recommendations.
    
    Returns personalized recommendations based on:
    - Price predictions
    - Sentiment analysis
    - Anomaly detection
    - Risk profile
    """
    # Load agent from database
    agent = await get_or_create_agent_from_db(portfolio_id, session)
    
    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # Generate recommendations
    recommendations = await agent.get_daily_recommendations(
        session=session,
        top_n=top_n,
        symbols=symbol_list,
        prediction_service=prediction_svc
    )
    
    return [rec.to_dict() for rec in recommendations]


async def get_or_create_agent_from_db(
    portfolio_id: str,
    session: AsyncSession
) -> DecisionAgent:
    """Load or create agent from database."""
    try:
        # Try to load existing portfolio
        portfolio_uuid = UUID(portfolio_id)
        portfolio_mgr = await portfolio_repo.load_portfolio(session, portfolio_uuid)
        
        if portfolio_mgr:
            # Create agent with loaded portfolio
            agent = DecisionAgent(
                portfolio_id=portfolio_uuid,
                risk_profile=portfolio_mgr.risk_profile,
                initial_capital=portfolio_mgr.initial_capital
            )
            agent.portfolio = portfolio_mgr  # Replace with loaded state
            return agent
        
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid portfolio ID")


@router.post("/portfolio/{portfolio_id}/trade")
async def execute_trade(
    portfolio_id: str,
    request: TradeRequest,
    session: AsyncSession = Depends(get_session)  # ✅ INJECT SESSION
):
    """Execute a trade with risk management."""
    agent = await get_or_create_agent_from_db(portfolio_id, session)
    
    try:
        result = await agent.execute_trade(
            session=session,
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            price=request.price,
            generate_explanation=request.generate_explanation
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Persist portfolio state
        await portfolio_repo.save_portfolio(session, agent.portfolio)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 4. Add Performance Visualization

### Step 4.1: New Endpoint

**File**: `app/ai/router.py` (ADD)

```python
@router.get("/portfolio/{portfolio_id}/performance/chart")
async def get_performance_chart(
    portfolio_id: str,
    session: AsyncSession = Depends(get_session),
    days: int = Query(default=30, ge=1, le=365, description="Number of days to retrieve")
):
    """
    Get time-series data for portfolio performance visualization.
    
    Returns data for creating performance charts:
    - Portfolio value over time
    - ROI percentage
    - Benchmark comparison (TUNINDEX)
    """
    try:
        portfolio_uuid = UUID(portfolio_id)
        
        # Get portfolio snapshots
        snapshots = await portfolio_repo.get_portfolio_snapshots(
            session, portfolio_uuid, limit=days
        )
        
        if not snapshots:
            return {
                "dates": [],
                "values": [],
                "roi": [],
                "message": "No historical data available"
            }
        
        # Extract data
        dates = [s["timestamp"] for s in snapshots]
        values = [s["total_value"] for s in snapshots]
        
        # Calculate ROI for each point
        agent = await get_or_create_agent_from_db(portfolio_id, session)
        initial = agent.portfolio.initial_capital
        roi = [((v - initial) / initial * 100) for v in values]
        
        # TODO: Add benchmark data (TUNINDEX)
        # For now, return portfolio data only
        
        return {
            "dates": dates,
            "values": values,
            "roi": roi,
            "initial_capital": initial,
            "current_value": values[-1] if values else 0,
            "total_return_pct": roi[-1] if roi else 0
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid portfolio ID")
    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 5. Add Rate Limiting

### Step 5.1: Install and Configure

**File**: `requirements.txt` (ADD)

```
slowapi==0.1.9
```

**File**: `app/ai/router.py` (UPDATE)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Add rate limit error handler to app
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."}
    )

# Apply to endpoints
@router.get("/recommendations")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def get_daily_recommendations(
    request: Request,  # Required for rate limiting
    session: AsyncSession = Depends(get_session),
    ...
):
    pass

@router.post("/portfolio/{portfolio_id}/trade")
@limiter.limit("20/minute")  # Max 20 trades per minute
async def execute_trade(
    request: Request,
    portfolio_id: str,
    ...
):
    pass
```

---

## 6. Add Caching to Aggregator

### Step 6.1: Simple In-Memory Cache

**File**: `app/ai/aggregator.py` (UPDATE)

```python
from datetime import datetime, timedelta
from typing import Any, Optional, Dict

class DataAggregator:
    def __init__(self):
        self.sentiment_analyzer = None
        self._cache: Dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _cache_key(self, prefix: str, symbol: str) -> str:
        """Generate cache key."""
        return f"{prefix}:{symbol}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still fresh."""
        if key not in self._cache:
            return False
        
        cached_time, _ = self._cache[key]
        return datetime.now() - cached_time < self._cache_ttl
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if self._is_cache_valid(key):
            _, data = self._cache[key]
            logger.debug(f"Cache hit: {key}")
            return data
        return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self._cache[key] = (datetime.now(), data)
    
    async def get_signals(
        self,
        symbol: str,
        session: AsyncSession,
        prediction_service = None
    ) -> MarketSignals:
        """Get signals with caching."""
        cache_key = self._cache_key("signals", symbol)
        
        # Check cache first
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Fetch fresh data
        signals = await self._fetch_signals_fresh(
            symbol, session, prediction_service
        )
        
        # Cache result
        self._set_cache(cache_key, signals)
        
        return signals
    
    async def _fetch_signals_fresh(
        self,
        symbol: str,
        session: AsyncSession,
        prediction_service
    ) -> MarketSignals:
        """Fetch fresh signals (original get_signals logic)."""
        # Move your current get_signals implementation here
        pass
```

---

## 7. Add Tests

### Step 7.1: Portfolio Tests

**File**: `tests/test_ai_portfolio.py` (NEW)

```python
import pytest
from datetime import date
from app.ai.portfolio import PortfolioManager, Position
from app.ai.profile import RiskProfile


def test_portfolio_initialization():
    """Test portfolio creation."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=10000.0
    )
    
    assert portfolio.initial_capital == 10000.0
    assert portfolio.cash_balance == 10000.0
    assert portfolio.total_value == 10000.0
    assert len(portfolio.positions) == 0


def test_buy_success():
    """Test successful buy order."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=10000.0
    )
    
    success, message = portfolio.buy("AMEN", 10, 12.50)
    
    assert success is True
    assert "AMEN" in portfolio.positions
    assert portfolio.positions["AMEN"].quantity == 10
    assert portfolio.cash_balance == 10000.0 - (10 * 12.50)


def test_buy_insufficient_funds():
    """Test buy rejection due to insufficient funds."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=100.0
    )
    
    success, message = portfolio.buy("AMEN", 100, 12.50)
    
    assert success is False
    assert "insuffisants" in message.lower()


def test_sell_success():
    """Test successful sell order."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=10000.0
    )
    
    # Buy first
    portfolio.buy("AMEN", 10, 10.0)
    
    # Sell at profit
    success, message, pnl = portfolio.sell("AMEN", 10, 12.0)
    
    assert success is True
    assert pnl == (12.0 - 10.0) * 10  # Profit of 20 TND
    assert "AMEN" not in portfolio.positions


def test_stop_loss_trigger():
    """Test stop-loss detection."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=10000.0
    )
    
    portfolio.buy("AMEN", 10, 100.0)
    portfolio.update_prices({"AMEN": 91.0})  # 9% loss
    
    triggered = portfolio.check_stop_losses()
    
    # Moderate profile has 8% stop-loss, so should trigger
    assert "AMEN" in triggered


def test_risk_limit_position_size():
    """Test position size limit enforcement."""
    portfolio = PortfolioManager(
        risk_profile=RiskProfile.CONSERVATIVE,  # Max 10% per position
        initial_capital=10000.0
    )
    
    # Try to buy 20% of portfolio (should fail)
    success, message = portfolio.buy("AMEN", 200, 10.0)
    
    assert success is False
    assert "position trop importante" in message.lower()
```

### Step 7.2: Metrics Tests

**File**: `tests/test_ai_metrics.py` (NEW)

```python
import pytest
from app.ai.metrics import MetricsCalculator


def test_roi_calculation():
    """Test ROI calculation."""
    calc = MetricsCalculator()
    
    roi = calc.calculate_roi(
        initial_capital=10000.0,
        current_value=11000.0
    )
    
    assert roi == 10.0  # 10% return


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    calc = MetricsCalculator()
    
    # Positive returns
    returns = [0.01, 0.02, 0.015, 0.02, 0.01]
    sharpe = calc.calculate_sharpe_ratio(returns)
    
    assert sharpe > 0  # Should be positive for positive returns


def test_max_drawdown():
    """Test maximum drawdown calculation."""
    calc = MetricsCalculator()
    
    # Portfolio values: rise, then fall
    values = [10000, 11000, 12000, 10500, 9000]
    max_dd = calc.calculate_max_drawdown(values)
    
    # Max drawdown from 12000 to 9000 = -25%
    assert max_dd == -25.0


def test_trade_statistics():
    """Test trade statistics calculation."""
    calc = MetricsCalculator()
    
    trades = [
        {"profit_loss": 100},
        {"profit_loss": 50},
        {"profit_loss": -30},
        {"profit_loss": -20}
    ]
    
    stats = calc.calculate_trade_statistics(trades)
    
    assert stats["total_trades"] == 4
    assert stats["winning_trades"] == 2
    assert stats["losing_trades"] == 2
    assert stats["win_rate"] == 50.0
```

---

## Summary

This document provides **ready-to-implement code** for:

1. ✅ Database persistence (models, migration, repository)
2. ✅ Complete data aggregator implementation
3. ✅ Fixed API router with dependency injection
4. ✅ Performance visualization endpoint
5. ✅ Rate limiting
6. ✅ Caching layer
7. ✅ Unit tests

**Next Steps**:
1. Copy code snippets to respective files
2. Run migration: `psql -d fixtrade -f db/002_portfolio_tables.sql`
3. Install dependencies: `pip install slowapi`
4. Run tests: `pytest tests/test_ai_*.py`
5. Test endpoints with Postman/curl

**Estimated Implementation Time**: 3-4 days
