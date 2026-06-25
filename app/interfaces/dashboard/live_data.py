"""Read the dashboard's persisted outputs from PostgreSQL."""

from __future__ import annotations

import os
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor

from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)
from app.interfaces.dashboard.schemas import CurrentMarketSnapshot, MarketSnapshot


def _connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg2.connect(database_url, connect_timeout=2)


def load_persisted_dashboard(symbol: str) -> dict:
    """Return the latest persisted market and pipeline results for a symbol."""
    with _connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT seance, cloture
            FROM stock_prices
            WHERE symbol=%s
            ORDER BY seance DESC
            LIMIT 45
            """,
            (symbol,),
        )
        price_rows = list(reversed(cur.fetchall()))
        cur.execute(
            """
            SELECT seance, cloture, quantite_negociee
            FROM stock_prices
            WHERE symbol=%s
            ORDER BY seance DESC
            LIMIT 20
            """,
            (symbol,),
        )
        snapshot_rows = cur.fetchall()

        cur.execute(
            """
            WITH latest_market AS (
                SELECT seance AS latest_date, cloture AS latest_close
                FROM stock_prices
                WHERE symbol=%s
                ORDER BY seance DESC
                LIMIT 1
            )
            SELECT DISTINCT ON (pp.target_date)
                pp.symbol, pp.target_date, pp.predicted_close,
                pp.confidence_lower, pp.confidence_upper
            FROM price_predictions pp
            CROSS JOIN latest_market lm
            WHERE pp.symbol=%s
              AND pp.target_date > lm.latest_date
              AND pp.predicted_close > 0
              AND pp.predicted_close BETWEEN
                  lm.latest_close * 0.65 AND lm.latest_close * 1.35
            ORDER BY pp.target_date, pp.created_at DESC
            LIMIT 5
            """,
            (symbol, symbol),
        )
        prediction_rows = cur.fetchall()

        cur.execute(
            """
            SELECT symbol, score_date, score, sentiment, article_count
            FROM sentiment_scores
            WHERE symbol=%s
            ORDER BY score_date DESC
            LIMIT 1
            """,
            (symbol,),
        )
        sentiment_row = cur.fetchone()

        cur.execute(
            """
            SELECT id, symbol, detected_at, anomaly_type, severity, description
            FROM anomaly_alerts
            WHERE symbol=%s
            ORDER BY detected_at DESC
            LIMIT 10
            """,
            (symbol,),
        )
        anomaly_rows = cur.fetchall()

        cur.execute(
            """
            SELECT symbol, action, confidence, reasoning
            FROM trade_recommendations
            WHERE symbol=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (symbol,),
        )
        recommendation_row = cur.fetchone()

    return {
        "current_snapshot": (
            CurrentMarketSnapshot(
                date=snapshot_rows[0]["seance"],
                close=snapshot_rows[0]["cloture"],
                previous_close=(
                    snapshot_rows[1]["cloture"]
                    if len(snapshot_rows) > 1
                    else snapshot_rows[0]["cloture"]
                ),
                volume=int(snapshot_rows[0]["quantite_negociee"] or 0),
                average_volume=int(
                    sum(int(row["quantite_negociee"] or 0) for row in snapshot_rows)
                    / len(snapshot_rows)
                ),
            )
            if snapshot_rows
            else None
        ),
        "historical_prices": [
            HistoricalPriceItem(date=row["seance"], close=row["cloture"])
            for row in price_rows
        ],
        "price_predictions": [
            PredictPriceItem(
                symbol=row["symbol"],
                target_date=row["target_date"],
                predicted_close=row["predicted_close"],
                confidence_lower=row["confidence_lower"]
                or row["predicted_close"],
                confidence_upper=row["confidence_upper"]
                or row["predicted_close"],
            )
            for row in prediction_rows
        ],
        "sentiment": (
            SentimentResponse(
                symbol=sentiment_row["symbol"],
                date=sentiment_row["score_date"],
                score=sentiment_row["score"],
                sentiment=sentiment_row["sentiment"],
                article_count=sentiment_row["article_count"],
            )
            if sentiment_row
            else None
        ),
        "anomalies": [
            AnomalyItem(
                id=row["id"],
                symbol=row["symbol"],
                detected_at=row["detected_at"],
                anomaly_type=row["anomaly_type"],
                severity=Decimal(str(row["severity"])),
                description=row["description"] or "",
            )
            for row in anomaly_rows
        ],
        "recommendation": (
            RecommendationResponse(
                symbol=recommendation_row["symbol"],
                action=recommendation_row["action"],
                confidence=recommendation_row["confidence"],
                reasoning=recommendation_row["reasoning"] or "",
            )
            if recommendation_row
            else None
        ),
    }


def load_pipeline_status() -> dict:
    """Return row counts and the most recent automation runs."""
    with _connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM stock_prices) AS prices,
                (SELECT COUNT(*) FROM scraped_articles) AS articles,
                (SELECT COUNT(*) FROM price_predictions) AS predictions,
                (SELECT COUNT(*) FROM anomaly_alerts) AS anomalies,
                (SELECT MAX(seance) FROM stock_prices) AS latest_market_date
            """
        )
        counts = dict(cur.fetchone())
        cur.execute(
            """
            SELECT job_name, status, started_at, finished_at, details, error
            FROM pipeline_runs
            ORDER BY started_at DESC
            LIMIT 10
            """
        )
        runs = [dict(row) for row in cur.fetchall()]
    return {"counts": counts, "recent_runs": runs}


def load_market_universe(limit: int = 60) -> list[MarketSnapshot]:
    """Load actively traded BVMT companies with their latest signals."""
    with _connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH global_latest AS (
                SELECT MAX(seance) AS latest_date FROM stock_prices
            ),
            ranked AS (
                SELECT
                    symbol,
                    seance,
                    cloture,
                    quantite_negociee,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY seance DESC
                    ) AS row_number
                FROM stock_prices
            ),
            price_summary AS (
                SELECT
                    symbol,
                    MAX(seance) FILTER (WHERE row_number = 1) AS latest_date,
                    MAX(cloture) FILTER (WHERE row_number = 1) AS close,
                    MAX(cloture) FILTER (WHERE row_number = 2) AS previous_close,
                    MAX(quantite_negociee) FILTER (WHERE row_number = 1) AS volume
                FROM ranked
                WHERE row_number <= 2
                GROUP BY symbol
            ),
            average_volume AS (
                SELECT symbol, AVG(quantite_negociee)::bigint AS avg_volume
                FROM (
                    SELECT
                        symbol,
                        quantite_negociee,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol ORDER BY seance DESC
                        ) AS row_number
                    FROM stock_prices
                ) recent
                WHERE row_number <= 20
                GROUP BY symbol
            ),
            latest_sentiment AS (
                SELECT DISTINCT ON (symbol) symbol, score
                FROM sentiment_scores
                ORDER BY symbol, score_date DESC
            ),
            latest_recommendation AS (
                SELECT DISTINCT ON (symbol)
                    symbol, action, confidence
                FROM trade_recommendations
                ORDER BY symbol, created_at DESC
            ),
            recent_anomalies AS (
                SELECT symbol, COUNT(*)::int AS anomaly_count
                FROM anomaly_alerts
                WHERE detected_at >= NOW() - INTERVAL '30 days'
                GROUP BY symbol
            )
            SELECT
                prices.symbol,
                prices.latest_date,
                prices.close,
                COALESCE(prices.previous_close, prices.close) AS previous_close,
                prices.close - COALESCE(prices.previous_close, prices.close) AS change,
                CASE
                    WHEN prices.previous_close > 0
                    THEN (
                        (prices.close - prices.previous_close)
                        / prices.previous_close * 100
                    )
                    ELSE 0
                END AS change_percent,
                COALESCE(prices.volume, 0) AS volume,
                COALESCE(vol.avg_volume, 0) AS average_volume,
                COALESCE(sentiment.score, 0) AS sentiment_score,
                recommendation.action AS recommendation,
                COALESCE(recommendation.confidence, 0) AS recommendation_confidence,
                COALESCE(anomalies.anomaly_count, 0) AS anomaly_count
            FROM price_summary prices
            CROSS JOIN global_latest global_dates
            LEFT JOIN average_volume vol ON vol.symbol = prices.symbol
            LEFT JOIN latest_sentiment sentiment ON sentiment.symbol = prices.symbol
            LEFT JOIN latest_recommendation recommendation
                ON recommendation.symbol = prices.symbol
            LEFT JOIN recent_anomalies anomalies ON anomalies.symbol = prices.symbol
            WHERE prices.close > 0
              AND prices.latest_date >= global_dates.latest_date - INTERVAL '14 days'
              AND prices.symbol !~* '(DS |DA |DROIT|OBLIG|BOND|TUNINDEX|USD|EUR)'
            ORDER BY COALESCE(vol.avg_volume, 0) DESC, prices.symbol
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [
        MarketSnapshot(
            symbol=row["symbol"],
            date=row["latest_date"],
            close=row["close"],
            previous_close=row["previous_close"],
            change=row["change"],
            change_percent=row["change_percent"],
            volume=row["volume"],
            average_volume=row["average_volume"],
            sentiment_score=row["sentiment_score"],
            recommendation=row["recommendation"],
            recommendation_confidence=row["recommendation_confidence"],
            anomaly_count=row["anomaly_count"],
        )
        for row in rows
    ]
