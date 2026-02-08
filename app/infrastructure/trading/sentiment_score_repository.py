"""
Adapter: Aggregated sentiment score repository.

Implements SentimentScoreRepository port.
Reads/writes the sentiment_scores table (daily aggregated per symbol).
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.domain.trading.entities import Sentiment, SentimentScore
from app.domain.trading.ports import SentimentScoreRepository

logger = logging.getLogger(__name__)


def _label_to_sentiment(label: str) -> Sentiment:
    """Map a DB string to Sentiment enum."""
    return Sentiment(label)


class SentimentScoreRepositoryAdapter(SentimentScoreRepository):
    """PostgreSQL adapter for the sentiment_scores table."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, score: SentimentScore) -> None:
        """Upsert an aggregated daily sentiment score."""
        query = text(
            """
            INSERT INTO sentiment_scores (symbol, score_date, score, sentiment, article_count)
            VALUES (:symbol, :score_date, :score, :sentiment, :article_count)
            ON CONFLICT (symbol, score_date)
            DO UPDATE SET
                score = EXCLUDED.score,
                sentiment = EXCLUDED.sentiment,
                article_count = EXCLUDED.article_count,
                created_at = NOW()
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                query,
                {
                    "symbol": score.symbol,
                    "score_date": score.date,
                    "score": float(score.score),
                    "sentiment": score.sentiment.value,
                    "article_count": score.article_count,
                },
            )
        logger.debug(
            "Saved sentiment score: symbol=%s date=%s score=%s",
            score.symbol,
            score.date,
            score.score,
        )

    def get(self, symbol: str, score_date: date) -> Optional[SentimentScore]:
        """Return the aggregated score for a symbol on a date, or None."""
        query = text(
            """
            SELECT symbol, score_date, score, sentiment, article_count
            FROM sentiment_scores
            WHERE symbol = :symbol AND score_date = :score_date
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(
                query, {"symbol": symbol, "score_date": score_date}
            ).fetchone()

        if not row:
            return None

        return SentimentScore(
            symbol=row[0],
            date=row[1],
            score=Decimal(str(row[2])),
            sentiment=_label_to_sentiment(row[3]),
            article_count=row[4],
        )

    def get_range(
        self, symbol: str, start: date, end: date
    ) -> list[SentimentScore]:
        """Return aggregated scores for a symbol within a date range."""
        query = text(
            """
            SELECT symbol, score_date, score, sentiment, article_count
            FROM sentiment_scores
            WHERE symbol = :symbol
              AND score_date >= :start
              AND score_date <= :end
            ORDER BY score_date ASC
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                query, {"symbol": symbol, "start": start, "end": end}
            ).fetchall()

        return [
            SentimentScore(
                symbol=r[0],
                date=r[1],
                score=Decimal(str(r[2])),
                sentiment=_label_to_sentiment(r[3]),
                article_count=r[4],
            )
            for r in rows
        ]
