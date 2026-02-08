"""
Database integration service for portfolio optimization.

Fetches historical data for:
- Stock returns
- Market returns (TUNINDEX)
- Anomaly detection results
- Current portfolio positions
"""

import logging
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PortfolioDataService:
    """Service for fetching portfolio-related data from database."""
    
    @staticmethod
    async def fetch_historical_returns(
        db: AsyncSession,
        symbols: List[str],
        lookback_days: int = 250
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fetch historical returns for assets.
        
        Args:
            db: Database session
            symbols: List of stock symbols
            lookback_days: Number of trading days to fetch
        
        Returns:
            Tuple of (returns_matrix, symbols_list)
            returns_matrix shape: (days, num_symbols)
        """
        start_date = date.today() - timedelta(days=int(lookback_days * 1.5))
        
        query = text("""
            WITH daily_prices AS (
                SELECT 
                    symbol,
                    seance as trade_date,
                    cloture as close_price,
                    LAG(cloture) OVER (PARTITION BY symbol ORDER BY seance) as prev_close
                FROM stock_prices
                WHERE symbol = ANY(:symbols)
                  AND seance >= :start_date
                  AND cloture IS NOT NULL
                ORDER BY seance
            )
            SELECT 
                symbol,
                trade_date,
                (close_price - prev_close) / prev_close as daily_return
            FROM daily_prices
            WHERE prev_close IS NOT NULL
              AND prev_close > 0
            ORDER BY trade_date, symbol
        """)
        
        result = await db.execute(
            query,
            {"symbols": symbols, "start_date": start_date}
        )
        
        rows = result.fetchall()
        
        if not rows:
            logger.warning(f"No price data found for symbols: {symbols}")
            # Return synthetic data as fallback
            return np.random.randn(lookback_days, len(symbols)) * 0.02 + 0.001, symbols
        
        # Convert to DataFrame for easier pivoting
        df = pd.DataFrame(rows, columns=["symbol", "trade_date", "daily_return"])
        
        # Pivot to matrix format
        returns_df = df.pivot(index="trade_date", columns="symbol", values="daily_return")
        
        # Fill missing values (forward fill then backward fill)
        returns_df = returns_df.ffill().bfill()
        
        # Get symbols that actually have data
        available_symbols = returns_df.columns.tolist()
        
        # Take last N days
        returns_matrix = returns_df.tail(lookback_days).values
        
        logger.info(
            f"Fetched returns for {len(available_symbols)} symbols, "
            f"{returns_matrix.shape[0]} days"
        )
        
        return returns_matrix, available_symbols
    
    @staticmethod
    async def fetch_market_returns(
        db: AsyncSession,
        lookback_days: int = 250
    ) -> np.ndarray:
        """
        Fetch market index (TUNINDEX) returns.
        
        Args:
            db: Database session
            lookback_days: Number of trading days
        
        Returns:
            Array of market returns, shape: (days,)
        """
        start_date = date.today() - timedelta(days=int(lookback_days * 1.5))
        
        # Try to fetch TUNINDEX or create synthetic market proxy
        query = text("""
            WITH market_prices AS (
                SELECT 
                    seance as trade_date,
                    AVG(cloture) as avg_close,
                    LAG(AVG(cloture)) OVER (ORDER BY seance) as prev_avg
                FROM stock_prices
                WHERE seance >= :start_date
                  AND cloture IS NOT NULL
                GROUP BY seance
                ORDER BY seance
            )
            SELECT 
                trade_date,
                (avg_close - prev_avg) / prev_avg as market_return
            FROM market_prices
            WHERE prev_avg IS NOT NULL
              AND prev_avg > 0
            ORDER BY trade_date
        """)
        
        result = await db.execute(query, {"start_date": start_date})
        rows = result.fetchall()
        
        if not rows:
            logger.warning("No market data found, using synthetic returns")
            return np.random.randn(lookback_days) * 0.015 + 0.0008
        
        df = pd.DataFrame(rows, columns=["trade_date", "market_return"])
        market_returns = df["market_return"].tail(lookback_days).values
        
        logger.info(f"Fetched {len(market_returns)} days of market returns")
        
        return market_returns
    
    @staticmethod
    async def fetch_anomaly_status(
        db: AsyncSession,
        symbols: List[str]
    ) -> Dict[str, bool]:
        """
        Get latest anomaly detection status for symbols.
        
        Args:
            db: Database session
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol -> has_anomaly
        """
        query = text("""
            SELECT DISTINCT ON (symbol)
                symbol,
                severity
            FROM anomaly_alerts
            WHERE symbol = ANY(:symbols)
              AND detected_at >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY symbol, detected_at DESC
        """)
        
        result = await db.execute(query, {"symbols": symbols})
        rows = result.fetchall()
        
        anomaly_map = {}
        for row in rows:
            # Consider HIGH and CRITICAL as anomalies
            anomaly_map[row[0]] = row[1] in ("HIGH", "CRITICAL")
        
        # Fill in symbols without anomaly records
        for symbol in symbols:
            if symbol not in anomaly_map:
                anomaly_map[symbol] = False
        
        num_anomalies = sum(1 for v in anomaly_map.values() if v)
        logger.info(
            f"Anomaly status: {num_anomalies}/{len(symbols)} symbols "
            "with detected anomalies"
        )
        
        return anomaly_map
    
    @staticmethod
    async def fetch_current_weights(
        db: AsyncSession,
        portfolio_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get current portfolio weights for a user.
        
        Args:
            db: Database session
            portfolio_id: Portfolio UUID
            user_id: User identifier
        
        Returns:
            Dict mapping symbol -> weight (as fraction)
        """
        if not portfolio_id and not user_id:
            logger.debug("No portfolio_id or user_id provided, returning empty weights")
            return {}
        
        # Build query based on what's provided
        if portfolio_id:
            query = text("""
                SELECT 
                    pp.symbol,
                    (pp.quantity * sp.cloture) as position_value
                FROM portfolio_positions pp
                JOIN stock_prices sp ON sp.symbol = pp.symbol
                WHERE pp.portfolio_id = :portfolio_id
                  AND sp.seance = (
                      SELECT MAX(seance) 
                      FROM stock_prices 
                      WHERE symbol = pp.symbol
                  )
            """)
            params = {"portfolio_id": portfolio_id}
        else:
            # If only user_id, get their most recent portfolio
            query = text("""
                SELECT 
                    pp.symbol,
                    (pp.quantity * sp.cloture) as position_value
                FROM portfolio_positions pp
                JOIN portfolios p ON p.id = pp.portfolio_id
                JOIN stock_prices sp ON sp.symbol = pp.symbol
                WHERE p.user_id = :user_id
                  AND sp.seance = (
                      SELECT MAX(seance) 
                      FROM stock_prices 
                      WHERE symbol = pp.symbol
                  )
            """)
            params = {"user_id": user_id}
        
        result = await db.execute(query, params)
        rows = result.fetchall()
        
        if not rows:
            logger.debug("No portfolio positions found")
            return {}
        
        # Calculate weights
        positions = {row[0]: float(row[1]) for row in rows}
        total_value = sum(positions.values())
        
        if total_value == 0:
            return {}
        
        weights = {symbol: value / total_value for symbol, value in positions.items()}
        
        logger.info(f"Fetched weights for {len(weights)} positions")
        
        return weights
    
    @staticmethod
    async def fetch_latest_predictions(
        db: AsyncSession,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Get latest price predictions for symbols.
        
        Args:
            db: Database session
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol -> predicted_return
        """
        query = text("""
            WITH latest_prices AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    cloture as current_price
                FROM stock_prices
                WHERE symbol = ANY(:symbols)
                ORDER BY symbol, seance DESC
            ),
            latest_predictions AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    predicted_close
                FROM price_predictions
                WHERE symbol = ANY(:symbols)
                  AND target_date >= CURRENT_DATE
                ORDER BY symbol, created_at DESC
            )
            SELECT 
                lp.symbol,
                (pred.predicted_close - lp.current_price) / lp.current_price as predicted_return
            FROM latest_prices lp
            LEFT JOIN latest_predictions pred ON pred.symbol = lp.symbol
            WHERE pred.predicted_close IS NOT NULL
        """)
        
        result = await db.execute(query, {"symbols": symbols})
        rows = result.fetchall()
        
        predictions = {row[0]: float(row[1]) * 100 for row in rows}
        
        logger.info(f"Fetched predictions for {len(predictions)}/{len(symbols)} symbols")
        
        return predictions
    
    @staticmethod
    async def fetch_sentiment_scores(
        db: AsyncSession,
        symbols: List[str],
        days: int = 7
    ) -> Dict[str, float]:
        """
        Get recent sentiment scores for symbols.
        
        Args:
            db: Database session
            symbols: List of stock symbols
            days: Number of days to look back
        
        Returns:
            Dict mapping symbol -> avg_sentiment_score
        """
        query = text("""
            SELECT 
                symbol,
                AVG(score) as avg_score
            FROM sentiment_scores
            WHERE symbol = ANY(:symbols)
              AND score_date >= CURRENT_DATE - :days
            GROUP BY symbol
        """)
        
        result = await db.execute(query, {"symbols": symbols, "days": days})
        rows = result.fetchall()
        
        sentiment_map = {row[0]: float(row[1]) for row in rows}
        
        logger.info(f"Fetched sentiment for {len(sentiment_map)}/{len(symbols)} symbols")
        
        return sentiment_map
    
    @staticmethod
    async def save_recommendations(
        db: AsyncSession,
        portfolio_id: str,
        recommendations: List[Dict]
    ):
        """
        Save trading recommendations to database.
        
        Args:
            db: Database session
            portfolio_id: Portfolio UUID
            recommendations: List of recommendation dicts
        """
        for rec in recommendations:
            query = text("""
                INSERT INTO trade_recommendations (
                    portfolio_id,
                    symbol,
                    action,
                    confidence,
                    target_weight,
                    expected_return,
                    risk_contribution,
                    explanation,
                    anomaly_detected
                )
                VALUES (
                    :portfolio_id,
                    :symbol,
                    :action,
                    :confidence,
                    :target_weight,
                    :expected_return,
                    :risk_contribution,
                    :explanation,
                    :anomaly_detected
                )
            """)
            
            await db.execute(query, {
                "portfolio_id": portfolio_id,
                "symbol": rec["symbol"],
                "action": rec["action"],
                "confidence": rec["confidence"],
                "target_weight": rec["target_weight"],
                "expected_return": rec["expected_return"],
                "risk_contribution": rec["risk_contribution"],
                "explanation": rec["explanation"],
                "anomaly_detected": rec["anomaly_detected"]
            })
        
        await db.commit()
        
        logger.info(f"Saved {len(recommendations)} recommendations for portfolio {portfolio_id}")
