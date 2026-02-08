"""
Adapter: Sentiment analysis NLP pipeline.

Implements SentimentAnalysisPort.
Wraps the SentimentAnalyzer NLP service to provide both:
- Raw text analysis (for per-article sentiment)
- Aggregated sentiment retrieval (from DB, for symbol-level queries)
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from app.domain.trading.entities import Sentiment, SentimentScore
from app.domain.trading.ports import SentimentAnalysisPort
from app.nlp.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalysisAdapter(SentimentAnalysisPort):
    """Concrete adapter for NLP-based sentiment analysis.

    Wraps the SentimentAnalyzer for raw text inference and queries
    the article_sentiments table for aggregated symbol-level scores.
    """

    def __init__(
        self,
        engine: Optional[Engine] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        self._engine = engine
        self._analyzer = analyzer

    def _get_analyzer(self) -> SentimentAnalyzer:
        """Lazy-load the NLP model to avoid startup cost if unused."""
        if self._analyzer is None:
            logger.info("Lazy-loading SentimentAnalyzer…")
            self._analyzer = SentimentAnalyzer()
        return self._analyzer

    def analyze_text(self, text: str) -> int:
        """Run NLP inference on raw text and return a sentiment score.

        Args:
            text: Article text to analyze.

        Returns:
            1 for positive, -1 for negative, 0 for neutral.
        """
        return self._get_analyzer().analyze(text)

    def get_sentiment(
        self, symbol: str, target_date: Optional[date] = None
    ) -> SentimentScore:
        """Return aggregated sentiment score for a symbol on a given date.

        Queries the article_sentiments table joined with scraped_articles
        to compute an average score for articles published on the target date.

        Args:
            symbol: BVMT stock ticker.
            target_date: Date to query. Defaults to today.

        Returns:
            Aggregated SentimentScore entity.
        """
        if self._engine is None:
            raise RuntimeError(
                "SentimentAnalysisAdapter requires a DB engine for "
                "get_sentiment. Pass an engine to the constructor."
            )

        query_date = target_date or date.today()

        # First try: read from pre-aggregated sentiment_scores table
        # (populated by AggregateDailySentimentUseCase)
        pre_agg_query = sql_text(
            """
            SELECT score, sentiment, article_count
            FROM sentiment_scores
            WHERE symbol = :symbol AND score_date = :target_date
            """
        )

        with self._engine.connect() as conn:
            pre_row = conn.execute(
                pre_agg_query,
                {"symbol": symbol, "target_date": query_date},
            ).fetchone()

        if pre_row and pre_row[2] and pre_row[2] > 0:
            return SentimentScore(
                symbol=symbol,
                date=query_date,
                score=Decimal(str(round(float(pre_row[0]), 4))),
                sentiment=Sentiment(pre_row[1]),
                article_count=pre_row[2],
            )

        # Fallback: compute on-the-fly from article_sentiments
        # joined through article_symbols so we filter BY SYMBOL
        query = sql_text(
            """
            SELECT
                AVG(ase.sentiment_score) AS avg_score,
                COUNT(*) AS article_count
            FROM article_sentiments ase
            JOIN scraped_articles sa ON sa.id = ase.article_id
            JOIN article_symbols asym ON asym.article_id = sa.id
            WHERE asym.symbol = :symbol
              AND sa.published_at::date = :target_date
            """
        )

        with self._engine.connect() as conn:
            row = conn.execute(
                query, {"symbol": symbol, "target_date": query_date}
            ).fetchone()

        avg_score = row[0] if row and row[0] is not None else 0
        article_count = row[1] if row else 0

        sentiment = _avg_to_sentiment(avg_score)

        return SentimentScore(
            symbol=symbol,
            date=query_date,
            score=Decimal(str(round(avg_score, 4))),
            sentiment=sentiment,
            article_count=article_count,
        )


def _avg_to_sentiment(avg_score: float) -> Sentiment:
    """Map an average numeric score to a Sentiment enum."""
    if avg_score > 0.3:
        return Sentiment.POSITIVE
    if avg_score < -0.3:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL
