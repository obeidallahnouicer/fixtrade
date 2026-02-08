"""
Intraday tick data generator & known anomalies loader.

Provides two capabilities:

1. **Intraday Generator**: Synthesizes realistic 1-minute price ticks
   from existing daily OHLCV data. Uses a Geometric Brownian Motion
   (GBM) model constrained to match the daily open/high/low/close.
   This fills the gap until real tick data is available from BVMT.

2. **Known Anomalies Loader**: Loads the curated labeled anomaly
   dataset from CSV into the ``known_anomalies`` database table.

Usage:
    python db/load_intraday_and_labels.py --symbols BIAT SFBT BT --days 30
    python db/load_intraday_and_labels.py --load-labels-only
"""

import argparse
import csv
import logging
import math
import os
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "fixtrade")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")

KNOWN_ANOMALIES_CSV = Path(__file__).resolve().parent.parent / "data" / "known_anomalies.csv"

# BVMT trading hours: 09:00 – 14:30 (Tunis time, UTC+1)
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(14, 30)
MINUTES_PER_SESSION = 330  # 5.5 hours × 60


def get_connection():
    """Create a database connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


# ══════════════════════════════════════════════════════════════
# 1) Intraday tick generator
# ══════════════════════════════════════════════════════════════


def generate_1min_ticks(
    symbol: str,
    trading_date: date,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    daily_volume: int,
    seed: int | None = None,
) -> list[tuple]:
    """Generate synthetic 1-minute price ticks for one trading day.

    Uses a bridge process (constrained Brownian motion) so that:
        - First tick = open_price
        - Last tick = close_price
        - max(ticks) ≈ high_price
        - min(ticks) ≈ low_price
        - sum(volumes) = daily_volume

    Args:
        symbol: Stock ticker.
        trading_date: The calendar date.
        open_price, high_price, low_price, close_price: Daily OHLC.
        daily_volume: Total volume for the day.
        seed: Optional random seed for reproducibility.

    Returns:
        List of tuples: (symbol, timestamp, price, volume, tick_type)
    """
    if seed is not None:
        random.seed(seed)

    n = MINUTES_PER_SESSION
    if n < 2:
        return []

    # --- Build a Brownian bridge from open → close ---
    # drift = total log return spread over n steps
    log_open = math.log(max(open_price, 0.01))
    log_close = math.log(max(close_price, 0.01))
    drift_per_step = (log_close - log_open) / n

    # Volatility calibrated from intraday range
    intraday_range = (high_price - low_price) / max(open_price, 0.01)
    sigma = max(intraday_range / (2 * math.sqrt(n)), 0.0001)

    log_prices = [log_open]
    for i in range(1, n):
        remaining = n - i
        # Bridge: pull toward close
        target_drift = (log_close - log_prices[-1]) / remaining
        noise = random.gauss(0, sigma)
        log_prices.append(log_prices[-1] + target_drift + noise)

    # Force last = close
    log_prices.append(log_close)

    # Convert to prices
    raw_prices = [math.exp(lp) for lp in log_prices]

    # Scale to match high/low
    raw_max = max(raw_prices)
    raw_min = min(raw_prices)
    raw_range = raw_max - raw_min

    if raw_range > 0:
        target_range = high_price - low_price
        scaled = [
            low_price + (p - raw_min) / raw_range * target_range
            for p in raw_prices
        ]
    else:
        scaled = [open_price] * (n + 1)

    # Force endpoints
    scaled[0] = open_price
    scaled[-1] = close_price

    # --- Distribute volume with U-shape (heavy at open/close) ---
    weights = []
    for i in range(n + 1):
        frac = i / n
        # U-shape: higher at start and end
        w = 2.0 - 4.0 * frac * (1.0 - frac) + random.uniform(0.1, 0.5)
        weights.append(max(w, 0.05))

    total_w = sum(weights)
    volumes = [max(1, int(daily_volume * w / total_w)) for w in weights]

    # Adjust to match total
    diff = daily_volume - sum(volumes)
    if diff != 0 and len(volumes) > 0:
        volumes[0] += diff

    # --- Build result tuples ---
    ticks = []
    base_dt = datetime.combine(trading_date, MARKET_OPEN, tzinfo=timezone(timedelta(hours=1)))

    for i in range(min(len(scaled), len(volumes))):
        ts = base_dt + timedelta(minutes=i)
        price = round(scaled[i], 3)
        vol = max(0, volumes[i])
        ticks.append((symbol, ts, price, vol, "1min"))

    return ticks


def generate_and_load_intraday(
    symbols: list[str],
    days: int = 30,
    conn=None,
) -> dict[str, int]:
    """Generate intraday data for symbols and insert into DB.

    Fetches the last N days of daily OHLCV from stock_prices,
    generates 1-min ticks for each day, and bulk-inserts.

    Args:
        symbols: List of BVMT tickers.
        days: Number of recent trading days to generate.
        conn: Optional DB connection (created if None).

    Returns:
        Dict mapping symbol → number of ticks inserted.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    results = {}

    try:
        cur = conn.cursor()

        for symbol in symbols:
            # Fetch daily OHLCV
            cur.execute(
                """
                SELECT seance, ouverture, plus_haut, plus_bas, cloture, quantite_negociee
                FROM stock_prices
                WHERE symbol = %s
                ORDER BY seance DESC
                LIMIT %s
                """,
                (symbol, days),
            )
            rows = cur.fetchall()

            if not rows:
                logger.warning("No daily data for %s — skipping", symbol)
                results[symbol] = 0
                continue

            all_ticks = []
            for row in rows:
                seance, ouv, phaut, pbas, clot, qty = row
                if not all([ouv, phaut, pbas, clot]):
                    continue

                ticks = generate_1min_ticks(
                    symbol=symbol,
                    trading_date=seance,
                    open_price=float(ouv),
                    high_price=float(phaut),
                    low_price=float(pbas),
                    close_price=float(clot),
                    daily_volume=int(qty or 0),
                    seed=hash((symbol, str(seance))) % (2**31),
                )
                all_ticks.extend(ticks)

            if all_ticks:
                execute_values(
                    cur,
                    """
                    INSERT INTO intraday_ticks (symbol, tick_timestamp, price, volume, tick_type)
                    VALUES %s
                    ON CONFLICT (symbol, tick_timestamp, tick_type) DO NOTHING
                    """,
                    all_ticks,
                    page_size=5000,
                )
                conn.commit()

            results[symbol] = len(all_ticks)
            logger.info("%s: generated %d intraday ticks", symbol, len(all_ticks))

        cur.close()
    finally:
        if own_conn:
            conn.close()

    return results


# ══════════════════════════════════════════════════════════════
# 2) Known anomalies loader
# ══════════════════════════════════════════════════════════════


def load_known_anomalies(csv_path: str | Path | None = None, conn=None) -> int:
    """Load labeled anomalies from CSV into the known_anomalies table.

    Args:
        csv_path: Path to the CSV. Defaults to data/known_anomalies.csv.
        conn: Optional DB connection.

    Returns:
        Number of rows inserted.
    """
    csv_path = Path(csv_path) if csv_path else KNOWN_ANOMALIES_CSV
    if not csv_path.exists():
        logger.error("Known anomalies CSV not found: %s", csv_path)
        return 0

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    inserted = 0

    try:
        cur = conn.cursor()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Skip comment lines
                if row.get("date", "").startswith("#"):
                    continue
                try:
                    rows.append((
                        row["symbol"].strip(),
                        row["date"].strip(),
                        row["anomaly_type"].strip(),
                        float(row.get("severity", 0.5)),
                        row.get("description", "").strip(),
                        row.get("source", "manual").strip(),
                        row.get("verified", "true").strip().lower() == "true",
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("Skipping invalid row: %s — %s", row, e)
                    continue

        if rows:
            execute_values(
                cur,
                """
                INSERT INTO known_anomalies
                    (symbol, anomaly_date, anomaly_type, severity, description, source, verified)
                VALUES %s
                ON CONFLICT (symbol, anomaly_date, anomaly_type) DO NOTHING
                """,
                rows,
                page_size=500,
            )
            inserted = cur.rowcount
            conn.commit()

        cur.close()
        logger.info("Loaded %d known anomalies from %s", inserted, csv_path)

    finally:
        if own_conn:
            conn.close()

    return inserted


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = ["BIAT", "SFBT", "BT", "ATTIJARI BANK", "SAH"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Load intraday tick data & known anomalies into FixTrade DB."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to generate intraday data for",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of recent trading days (default: 30)",
    )
    parser.add_argument(
        "--load-labels-only",
        action="store_true",
        help="Only load known anomalies CSV, skip intraday generation",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip loading known anomalies, only generate intraday data",
    )

    args = parser.parse_args()

    # Load known anomalies
    if not args.skip_labels:
        count = load_known_anomalies()
        print(f"✓ Loaded {count} known anomalies into DB")

    # Generate intraday
    if not args.load_labels_only:
        results = generate_and_load_intraday(args.symbols, args.days)
        total = sum(results.values())
        print(f"\n✓ Generated intraday ticks:")
        for sym, cnt in results.items():
            print(f"    {sym:20s} → {cnt:>8,} ticks")
        print(f"    {'TOTAL':20s} → {total:>8,} ticks")


if __name__ == "__main__":
    main()
