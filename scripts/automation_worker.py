#!/usr/bin/env python3
"""Automated, database-first FixTrade data pipeline.

The worker is deliberately idempotent. It can safely run on startup and then
at a fixed interval:

1. Apply database migrations.
2. Backfill historical BVMT prices and fallback articles when required.
3. Link articles to symbols and calculate lightweight sentiment.
4. Persist five-session forecasts derived from actual market history.
5. Detect and persist price/volume anomalies.
6. Persist trading recommendations and pipeline run status.

The heavyweight training pipeline remains available as a weekly job, while
this worker guarantees that the dashboard always has fresh persisted outputs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import psycopg2
from dateutil import parser as date_parser
from psycopg2.extras import Json, RealDictCursor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INTERVAL_SECONDS = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "300"))
RETRAIN_INTERVAL_SECONDS = int(
    os.getenv("PIPELINE_RETRAIN_INTERVAL_SECONDS", "604800")
)
TRACKED_SYMBOLS = tuple(
    value.strip()
    for value in os.getenv("PIPELINE_SYMBOLS", "BIAT,SFBT,BT").split(",")
    if value.strip()
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("fixtrade.automation")

POSITIVE_WORDS = {
    "hausse", "croissance", "benefice", "bénéfice", "profit", "progression",
    "amélioration", "amelioration", "record", "succès", "succes", "positif",
    "gain", "dividende", "investissement", "partenariat", "augmentation",
}
NEGATIVE_WORDS = {
    "baisse", "perte", "déficit", "deficit", "recul", "crise", "risque",
    "négatif", "negatif", "chute", "dette", "sanction", "licenciement",
    "diminution", "faillite", "inflation",
}


def db_connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def wait_for_database(timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with db_connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return
        except Exception:
            logger.info("Waiting for PostgreSQL...")
            time.sleep(2)
    raise RuntimeError("PostgreSQL did not become available")


def apply_migrations() -> None:
    with db_connect() as conn, conn.cursor() as cur:
        for path in sorted((ROOT / "db").glob("[0-9][0-9][0-9]_*.sql")):
            logger.info("Applying migration %s", path.name)
            cur.execute(path.read_text(encoding="utf-8"))
        conn.commit()


def fail_stale_runs() -> None:
    """Close audit rows left behind by a killed/restarted worker."""
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status='failed',
                finished_at=NOW(),
                error=COALESCE(error, 'Worker stopped before completion')
            WHERE status='running'
            """
        )
        conn.commit()


@contextmanager
def recorded_run(job_name: str):
    run_id = None
    started = time.monotonic()
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (job_name, status) VALUES (%s, 'running') RETURNING id",
            (job_name,),
        )
        run_id = cur.fetchone()[0]
        conn.commit()
    details: dict[str, Any] = {}
    try:
        yield details
    except Exception as exc:
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_runs
                SET status='failed', finished_at=NOW(), details=%s, error=%s
                WHERE id=%s
                """,
                (Json(details), str(exc), run_id),
            )
            conn.commit()
        raise
    else:
        details["duration_seconds"] = round(time.monotonic() - started, 2)
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_runs
                SET status='completed', finished_at=NOW(), details=%s
                WHERE id=%s
                """,
                (Json(details), run_id),
            )
            conn.commit()


def table_count(table: str) -> int:
    allowed = {
        "stock_prices", "scraped_articles", "article_sentiments",
        "article_symbols", "price_predictions", "anomaly_alerts",
    }
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])


def latest_database_market_date() -> date | None:
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT MAX(seance) FROM stock_prices")
        return cur.fetchone()[0]


def latest_bundled_market_date() -> date | None:
    """Read the newest date from the most recent bundled BVMT file."""
    candidates = sorted((ROOT / "data" / "raw").glob("histo_cotation_*.csv"))
    if not candidates:
        return None
    newest: date | None = None
    with candidates[-1].open("r", encoding="utf-8-sig", errors="ignore") as handle:
        next(handle, None)
        for line in handle:
            first = line.split(";", 1)[0].strip()
            try:
                parsed = datetime.strptime(first, "%d/%m/%Y").date()
            except ValueError:
                continue
            if newest is None or parsed > newest:
                newest = parsed
    return newest


def run_python(script: str, *args: str, timeout: int = 3600) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    logger.info("Running %s", " ".join(command))
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    subprocess.run(
        command,
        cwd=ROOT,
        env=child_env,
        check=True,
        timeout=timeout,
    )


def run_incremental_etl() -> dict[str, Any]:
    """Run the real Bronze/Silver/Gold ETL, falling back to DB ingestion."""
    command = [sys.executable, "-m", "prediction", "etl", "--incremental"]
    try:
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        subprocess.run(
            command,
            cwd=ROOT,
            env=child_env,
            check=True,
            timeout=3600,
        )
        return {"status": "completed", "mode": "medallion"}
    except Exception as exc:
        logger.exception("Medallion ETL failed; direct DB ingestion remains available.")
        return {"status": "failed", "error": str(exc)}


def bootstrap_data() -> dict[str, int]:
    database_latest = latest_database_market_date()
    bundled_latest = latest_bundled_market_date()
    if database_latest is None or (
        bundled_latest is not None and database_latest < bundled_latest
    ):
        run_python("db/load_data.py")

    before_articles = table_count("scraped_articles")
    fallback_candidates = (
        ROOT / "data" / "scraped_fallback.jsonl",
        ROOT / "scraped_fallback.jsonl",
    )
    if before_articles == 0:
        fallback = next((path for path in fallback_candidates if path.exists()), None)
        if fallback:
            with db_connect() as conn, conn.cursor() as cur:
                for line in fallback.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    published_at = None
                    if item.get("date"):
                        try:
                            published_at = date_parser.parse(str(item["date"]))
                        except (TypeError, ValueError, OverflowError):
                            published_at = None
                    cur.execute(
                        """
                        INSERT INTO scraped_articles
                            (url, title, summary, content, published_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO NOTHING
                        """,
                        (
                            item.get("url"),
                            item.get("title"),
                            item.get("summary"),
                            item.get("content"),
                            published_at or None,
                        ),
                    )
                conn.commit()

    return {
        "prices": table_count("stock_prices"),
        "articles": table_count("scraped_articles"),
    }


def _sentiment_score(text: str) -> tuple[int, float]:
    normalized = text.lower()
    positive = sum(normalized.count(word) for word in POSITIVE_WORDS)
    negative = sum(normalized.count(word) for word in NEGATIVE_WORDS)
    total = positive + negative
    if total == 0:
        return 0, 0.5
    raw = (positive - negative) / total
    score = 1 if raw > 0.15 else -1 if raw < -0.15 else 0
    confidence = min(0.95, 0.55 + abs(raw) * 0.4)
    return score, confidence


def process_articles(batch_size: int = 1000) -> dict[str, int]:
    from app.domain.trading.article_symbol_matcher import ArticleSymbolMatcher

    matcher = ArticleSymbolMatcher()
    analyzed = linked = aggregated = 0
    with db_connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (sa.id)
                sa.id, sa.title, sa.summary, sa.content
            FROM scraped_articles sa
            LEFT JOIN article_sentiments sent ON sent.article_id = sa.id
            LEFT JOIN article_symbols sym ON sym.article_id = sa.id
            WHERE sent.id IS NULL OR sym.id IS NULL
            ORDER BY sa.id, sa.published_at DESC NULLS LAST
            LIMIT %s
            """,
            (batch_size,),
        )
        articles = cur.fetchall()

        for article in articles:
            text = " ".join(
                part for part in (
                    article["title"], article["summary"], article["content"]
                ) if part
            )
            score, confidence = _sentiment_score(text)
            label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            cur.execute(
                """
                INSERT INTO article_sentiments
                    (article_id, sentiment_label, sentiment_score, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id) DO NOTHING
                """,
                (article["id"], label, score, confidence),
            )
            analyzed += cur.rowcount

            for match in matcher.match(text):
                cur.execute(
                    """
                    INSERT INTO article_symbols
                        (article_id, symbol, match_method, confidence)
                    VALUES (%s, %s, 'keyword', %s)
                    ON CONFLICT (article_id, symbol) DO NOTHING
                    """,
                    (
                        article["id"],
                        match.symbol.upper(),
                        min(1.0, 0.7 + match.match_count * 0.1),
                    ),
                )
                linked += cur.rowcount

        cur.execute(
            """
            INSERT INTO sentiment_scores
                (symbol, score_date, score, sentiment, article_count)
            SELECT
                sym.symbol,
                COALESCE(sa.published_at::date, sa.created_at::date),
                AVG(sent.sentiment_score)::numeric(5,4),
                CASE
                    WHEN AVG(sent.sentiment_score) > 0.3 THEN 'positive'
                    WHEN AVG(sent.sentiment_score) < -0.3 THEN 'negative'
                    ELSE 'neutral'
                END,
                COUNT(*)
            FROM article_sentiments sent
            JOIN scraped_articles sa ON sa.id = sent.article_id
            JOIN article_symbols sym ON sym.article_id = sa.id
            GROUP BY sym.symbol, COALESCE(sa.published_at::date, sa.created_at::date)
            ON CONFLICT (symbol, score_date)
            DO UPDATE SET
                score=EXCLUDED.score,
                sentiment=EXCLUDED.sentiment,
                article_count=EXCLUDED.article_count,
                created_at=NOW()
            """
        )
        aggregated = cur.rowcount
        conn.commit()
    return {"analyzed": analyzed, "linked": linked, "aggregated": aggregated}


def available_symbols() -> list[str]:
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT symbol
            FROM stock_prices
            WHERE symbol = ANY(%s)
            """,
            (list(TRACKED_SYMBOLS),),
        )
        tracked_in_database = {row[0] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT symbol
            FROM stock_prices
            GROUP BY symbol
            HAVING COUNT(*) >= 20
            ORDER BY SUM(quantite_negociee) DESC NULLS LAST
            LIMIT 30
            """
        )
        database_symbols = [row[0] for row in cur.fetchall()]
    preferred = [
        symbol for symbol in TRACKED_SYMBOLS if symbol in tracked_in_database
    ]
    return preferred + [s for s in database_symbols if s not in preferred]


def _next_trading_days(start: date, count: int) -> Iterable[date]:
    current = start
    produced = 0
    while produced < count:
        current += timedelta(days=1)
        if current.weekday() < 5:
            produced += 1
            yield current


def refresh_predictions(symbols: list[str]) -> int:
    persisted = 0
    with db_connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for symbol in symbols:
            cur.execute(
                """
                SELECT seance, cloture
                FROM stock_prices
                WHERE symbol=%s
                ORDER BY seance DESC
                LIMIT 60
                """,
                (symbol,),
            )
            history = list(reversed(cur.fetchall()))
            if len(history) < 10:
                continue

            closes = [float(row["cloture"]) for row in history]
            window = closes[-20:]
            x_mean = (len(window) - 1) / 2
            y_mean = fmean(window)
            denominator = sum((i - x_mean) ** 2 for i in range(len(window)))
            slope = (
                sum((i - x_mean) * (value - y_mean) for i, value in enumerate(window))
                / denominator
                if denominator
                else 0.0
            )
            max_step = max(window[-1] * 0.02, 0.001)
            slope = max(-max_step, min(max_step, slope))
            volatility = pstdev(
                [
                    (window[i] - window[i - 1]) / window[i - 1]
                    for i in range(1, len(window))
                    if window[i - 1]
                ]
            ) if len(window) > 2 else 0.02
            spread = max(window[-1] * volatility * 1.96, window[-1] * 0.015)

            for horizon, target in enumerate(
                _next_trading_days(history[-1]["seance"], 5), start=1
            ):
                predicted = max(0.001, window[-1] + slope * horizon)
                cur.execute(
                    """
                    INSERT INTO price_predictions
                        (symbol, target_date, predicted_close, confidence_lower,
                         confidence_upper, confidence_score, model_name, horizon_days)
                    VALUES (%s, %s, %s, %s, %s, %s, 'auto_trend', %s)
                    ON CONFLICT (symbol, target_date, model_name)
                    DO UPDATE SET
                        predicted_close=EXCLUDED.predicted_close,
                        confidence_lower=EXCLUDED.confidence_lower,
                        confidence_upper=EXCLUDED.confidence_upper,
                        confidence_score=EXCLUDED.confidence_score,
                        horizon_days=EXCLUDED.horizon_days,
                        created_at=NOW()
                    """,
                    (
                        symbol, target, predicted, max(0.001, predicted - spread),
                        predicted + spread,
                        max(0.5, min(0.9, 1 - volatility * 5)),
                        horizon,
                    ),
                )
                persisted += 1
        conn.commit()
    return persisted


def refresh_trained_predictions(symbols: list[str]) -> int:
    """Run the trained ensemble when its runtime and artifacts are healthy."""
    if os.getenv("PIPELINE_ENABLE_ML", "true").lower() not in {"1", "true", "yes"}:
        return 0
    try:
        from prediction.inference import PredictionService

        service = PredictionService()
        produced = 0
        for symbol in symbols[:10]:
            service._cache.invalidate_predictions(symbol)
            results = service.predict(symbol=symbol, horizon_days=5)
            produced += sum(
                1
                for result in results
                if result.model_name != "fallback" and result.predicted_close > 0
            )
        return produced
    except Exception:
        logger.exception(
            "Trained inference unavailable; data-derived forecasts remain active."
        )
        return 0


def detect_anomalies(symbols: list[str]) -> int:
    persisted = 0
    with db_connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for symbol in symbols:
            cur.execute(
                """
                SELECT seance, cloture, quantite_negociee
                FROM stock_prices
                WHERE symbol=%s
                ORDER BY seance DESC
                LIMIT 30
                """,
                (symbol,),
            )
            rows = list(reversed(cur.fetchall()))
            if len(rows) < 10:
                continue
            latest = rows[-1]
            previous = rows[-2]
            volumes = [float(row["quantite_negociee"] or 0) for row in rows[:-1]]
            avg_volume = fmean(volumes) if volumes else 0
            price_move = (
                (float(latest["cloture"]) - float(previous["cloture"]))
                / float(previous["cloture"])
                if previous["cloture"]
                else 0
            )
            volume_ratio = (
                float(latest["quantite_negociee"] or 0) / avg_volume
                if avg_volume
                else 0
            )
            alerts: list[tuple[str, float, str]] = []
            if abs(price_move) >= 0.03:
                alerts.append(
                    (
                        "price_swing_daily",
                        min(1.0, abs(price_move) * 8),
                        f"Latest real close moved {price_move * 100:.1f}% versus the prior session.",
                    )
                )
            if volume_ratio >= 2.5:
                alerts.append(
                    (
                        "volume_spike",
                        min(1.0, volume_ratio / 5),
                        f"Latest real volume is {volume_ratio:.1f}x the previous 29-session average.",
                    )
                )

            for anomaly_type, severity, description in alerts:
                alert_id = uuid5(
                    NAMESPACE_URL,
                    f"fixtrade:{symbol}:{latest['seance']}:{anomaly_type}",
                )
                cur.execute(
                    """
                    INSERT INTO anomaly_alerts
                        (id, symbol, detected_at, anomaly_type, severity, description)
                    VALUES (%s, %s, NOW(), %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (str(alert_id), symbol, anomaly_type, severity, description),
                )
                persisted += cur.rowcount
        conn.commit()
    return persisted


def refresh_recommendations(symbols: list[str]) -> int:
    written = 0
    with db_connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for symbol in symbols:
            cur.execute(
                """
                WITH current_price AS (
                    SELECT cloture
                    FROM stock_prices WHERE symbol=%s
                    ORDER BY seance DESC LIMIT 1
                ), forecast AS (
                    SELECT predicted_close
                    FROM price_predictions
                    WHERE symbol=%s
                    ORDER BY target_date DESC, created_at DESC LIMIT 1
                ), sentiment AS (
                    SELECT score
                    FROM sentiment_scores
                    WHERE symbol=%s
                    ORDER BY score_date DESC LIMIT 1
                )
                SELECT
                    cp.cloture,
                    f.predicted_close,
                    COALESCE(s.score, 0) AS score
                FROM current_price cp
                CROSS JOIN forecast f
                LEFT JOIN sentiment s ON TRUE
                """,
                (symbol, symbol, symbol),
            )
            row = cur.fetchone()
            if not row or not row["cloture"]:
                continue
            expected = (
                float(row["predicted_close"]) - float(row["cloture"])
            ) / float(row["cloture"])
            combined = expected + float(row["score"]) * 0.01
            action = "buy" if combined > 0.015 else "sell" if combined < -0.015 else "hold"
            confidence = min(0.9, 0.55 + abs(combined) * 5)
            cur.execute(
                """
                INSERT INTO trade_recommendations
                    (symbol, action, confidence, reasoning)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    symbol,
                    action,
                    confidence,
                    (
                        f"Automated signal from persisted market data: "
                        f"five-session expected move {expected * 100:.1f}% "
                        f"with sentiment score {float(row['score']):.2f}."
                    ),
                ),
            )
            written += 1
        conn.commit()
    return written


def run_cycle() -> None:
    with recorded_run("automation_cycle") as details:
        database_latest_before = latest_database_market_date()
        bundled_latest = latest_bundled_market_date()
        market_data_changed = (
            database_latest_before is None
            or (
                bundled_latest is not None
                and database_latest_before < bundled_latest
            )
        )
        details["bootstrap"] = bootstrap_data()
        details["etl"] = (
            run_incremental_etl()
            if market_data_changed
            or os.getenv("PIPELINE_FORCE_ETL", "false").lower()
            in {"1", "true", "yes"}
            else {"status": "skipped", "reason": "no new market file"}
        )
        details["articles"] = process_articles()
        symbols = available_symbols()
        details["symbols"] = symbols
        details["trend_predictions"] = refresh_predictions(symbols)
        details["ml_predictions"] = refresh_trained_predictions(symbols[:3])
        details["anomalies"] = detect_anomalies(symbols)
        details["recommendations"] = refresh_recommendations(symbols)
        logger.info("Automation cycle complete: %s", json.dumps(details, default=str))


def maybe_retrain(last_retrain: float) -> float:
    if time.monotonic() - last_retrain < RETRAIN_INTERVAL_SECONDS:
        return last_retrain
    try:
        with recorded_run("weekly_retrain") as details:
            run_python(
                "run_training.py",
                "--skip-etl",
                "--final-only",
                "--no-ui",
                timeout=7200,
            )
            details["trained"] = True
        return time.monotonic()
    except Exception:
        logger.exception("Weekly model retraining failed; persisted trend forecasts remain active.")
        return time.monotonic()


def main() -> int:
    wait_for_database()
    apply_migrations()
    fail_stale_runs()
    last_retrain = time.monotonic()
    run_once = "--once" in sys.argv
    while True:
        try:
            run_cycle()
        except Exception:
            logger.exception("Automation cycle failed")
        if run_once:
            return 0
        last_retrain = maybe_retrain(last_retrain)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
