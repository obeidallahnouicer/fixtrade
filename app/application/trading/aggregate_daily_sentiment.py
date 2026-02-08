"""
Use case: Compute and persist daily aggregated sentiment scores.

For each symbol that has article_sentiments linked via article_symbols,
compute the average sentiment score on each date and upsert into the
sentiment_scores table.

This is the missing "Score de Sentiment Quotidien" aggregation pipeline.

Input:  AggregateDailySentimentCommand (symbol?, days_back)
Output: AggregateDailySentimentResult  (counts)
Side effects: Upserts rows in sentiment_scores table.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from app.domain.trading.entities import Sentiment, SentimentScore
from app.domain.trading.ports import SentimentScoreRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregateDailySentimentCommand:
    """Input DTO for daily sentiment aggregation.

    Attributes:
        symbol: Optional — if set, only aggregate for this symbol.
                If None, aggregate for all symbols that have linked articles.
        days_back: Number of past days to (re-)aggregate.
    """

    symbol: Optional[str] = None
    days_back: int = 30


@dataclass(frozen=True)
class DailyScoreItem:
    """A single aggregated score in the output."""

    symbol: str
    score_date: date
    score: Decimal
    sentiment: str
    article_count: int


@dataclass(frozen=True)
class AggregateDailySentimentResult:
    """Output DTO summarising an aggregation run."""

    symbols_processed: int
    dates_processed: int
    scores_upserted: int
    scores: list[DailyScoreItem]


def _avg_to_sentiment(avg_score: float) -> Sentiment:
    """Map an average numeric score to a Sentiment enum."""
    if avg_score > 0.3:
        return Sentiment.POSITIVE
    if avg_score < -0.3:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


class AggregateDailySentimentUseCase:
    """Computes daily aggregated sentiment scores per symbol.

    Reads from article_sentiments + article_symbols (via a raw query
    on the DB engine) and writes to sentiment_scores via the
    SentimentScoreRepository port.
    """

    def __init__(
        self,
        score_repo: SentimentScoreRepository,
        db_engine,
    ) -> None:
        """
        Args:
            score_repo: Repository to persist aggregated scores.
            db_engine: SQLAlchemy engine for the raw aggregation query.
        """
        self._score_repo = score_repo
        self._engine = db_engine

    def execute(
        self, command: AggregateDailySentimentCommand
    ) -> AggregateDailySentimentResult:
        """Run daily sentiment aggregation.

        Args:
            command: Contains optional symbol filter and days_back.

        Returns:
            Summary of the aggregation run.
        """
        from sqlalchemy import text as sql_text

        logger.info(
            "Aggregating daily sentiment: symbol=%s, days_back=%d",
            command.symbol or "ALL",
            command.days_back,
        )

        cutoff = date.today() - timedelta(days=command.days_back)

        # Build the aggregation query
        # Joins: article_sentiments → article_symbols → scraped_articles
        # Groups by: symbol, date
        base_query = """
            SELECT
                asym.symbol,
                sa.published_at::date AS score_date,
                AVG(ase.sentiment_score) AS avg_score,
                COUNT(*) AS article_count
            FROM article_sentiments ase
            JOIN scraped_articles sa ON sa.id = ase.article_id
            JOIN article_symbols asym ON asym.article_id = sa.id
            WHERE sa.published_at::date >= :cutoff
        """

        params: dict = {"cutoff": cutoff}

        if command.symbol:
            base_query += " AND asym.symbol = :symbol"
            params["symbol"] = command.symbol

        base_query += """
            GROUP BY asym.symbol, sa.published_at::date
            ORDER BY asym.symbol, score_date
        """

        with self._engine.connect() as conn:
            rows = conn.execute(sql_text(base_query), params).fetchall()

        if not rows:
            logger.info("No articles with symbol links found for aggregation.")
            return AggregateDailySentimentResult(
                symbols_processed=0,
                dates_processed=0,
                scores_upserted=0,
                scores=[],
            )

        symbols_seen: set[str] = set()
        dates_seen: set[date] = set()
        score_items: list[DailyScoreItem] = []

        for row in rows:
            sym = row[0]
            d = row[1]
            avg = float(row[2])
            count = row[3]

            sentiment = _avg_to_sentiment(avg)

            score_entity = SentimentScore(
                symbol=sym,
                date=d,
                score=Decimal(str(round(avg, 4))),
                sentiment=sentiment,
                article_count=count,
            )

            self._score_repo.save(score_entity)

            symbols_seen.add(sym)
            dates_seen.add(d)
            score_items.append(
                DailyScoreItem(
                    symbol=sym,
                    score_date=d,
                    score=score_entity.score,
                    sentiment=sentiment.value,
                    article_count=count,
                )
            )

        result = AggregateDailySentimentResult(
            symbols_processed=len(symbols_seen),
            dates_processed=len(dates_seen),
            scores_upserted=len(score_items),
            scores=score_items,
        )

        logger.info(
            "Aggregation complete: %d symbols, %d dates, %d scores upserted.",
            result.symbols_processed,
            result.dates_processed,
            result.scores_upserted,
        )

        return result
