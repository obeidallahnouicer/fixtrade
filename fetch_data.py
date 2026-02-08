"""
Helper script to fetch real stock data from database.
Used by Streamlit dashboard to show actual historical prices.
"""

import sys
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings


def get_historical_prices(symbol: str, days_back: int = 90) -> pd.DataFrame:
    """
    Fetch historical stock prices from database.
    
    Args:
        symbol: Stock symbol (e.g., 'BIAT')
        days_back: Number of days to fetch
    
    Returns:
        DataFrame with columns: date, open, close, high, low, volume
    """
    engine = create_engine(settings.database_url)
    
    query = text("""
        SELECT 
            seance as date,
            ouverture as open,
            cloture as close,
            plus_haut as high,
            plus_bas as low,
            quantite_negociee as volume,
            capitaux
        FROM stock_prices
        WHERE symbol = :symbol
        AND seance >= CURRENT_DATE - INTERVAL ':days days'
        ORDER BY seance ASC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol, "days": days_back})
        rows = result.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume', 'capitaux'])
    
    # Convert to float
    for col in ['open', 'close', 'high', 'low', 'capitaux']:
        df[col] = df[col].astype(float)
    
    return df


def get_latest_price(symbol: str) -> dict:
    """
    Get the most recent closing price for a symbol.
    
    Returns:
        dict with: symbol, date, close, volume
    """
    engine = create_engine(settings.database_url)
    
    query = text("""
        SELECT 
            symbol,
            seance as date,
            cloture as close,
            quantite_negociee as volume
        FROM stock_prices
        WHERE symbol = :symbol
        ORDER BY seance DESC
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol})
        row = result.fetchone()
    
    if not row:
        return {}
    
    return {
        "symbol": row[0],
        "date": row[1],
        "close": float(row[2]),
        "volume": int(row[3]),
    }


def get_all_symbols() -> list[str]:
    """Get list of all available stock symbols."""
    engine = create_engine(settings.database_url)
    
    query = text("""
        SELECT DISTINCT symbol
        FROM stock_prices
        ORDER BY symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        symbols = [row[0] for row in result.fetchall()]
    
    return symbols


def get_market_summary() -> dict:
    """
    Get market-wide summary statistics.
    
    Returns:
        dict with: total_symbols, total_volume, avg_price_change
    """
    engine = create_engine(settings.database_url)
    
    query = text("""
        WITH latest_date AS (
            SELECT MAX(seance) as max_date FROM stock_prices
        ),
        today_data AS (
            SELECT 
                symbol,
                cloture as close_today,
                quantite_negociee as volume
            FROM stock_prices
            WHERE seance = (SELECT max_date FROM latest_date)
        ),
        yesterday_data AS (
            SELECT 
                symbol,
                cloture as close_yesterday
            FROM stock_prices
            WHERE seance = (SELECT max_date FROM latest_date) - INTERVAL '1 day'
        )
        SELECT 
            COUNT(DISTINCT t.symbol) as total_symbols,
            SUM(t.volume) as total_volume,
            AVG((t.close_today - y.close_yesterday) / y.close_yesterday * 100) as avg_change_pct
        FROM today_data t
        LEFT JOIN yesterday_data y ON t.symbol = y.symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        row = result.fetchone()
    
    if not row:
        return {}
    
    return {
        "total_symbols": int(row[0]) if row[0] else 0,
        "total_volume": int(row[1]) if row[1] else 0,
        "avg_change_pct": float(row[2]) if row[2] else 0.0,
    }


def main():
    """CLI interface for testing."""
    if len(sys.argv) < 2:
        print("Usage: python fetch_data.py [symbol] [days_back]")
        print("Example: python fetch_data.py BIAT 90")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    
    print(f"Fetching {days_back} days of data for {symbol}...")
    df = get_historical_prices(symbol, days_back)
    
    if df.empty:
        print(f"No data found for {symbol}")
    else:
        print(f"\n{len(df)} rows found")
        print(f"\nLatest price: {df['close'].iloc[-1]:.3f} TND")
        print(f"Average volume: {df['volume'].mean():.0f}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())


if __name__ == "__main__":
    main()
