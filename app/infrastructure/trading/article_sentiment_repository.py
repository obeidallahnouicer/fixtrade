"""
Adapter: Article sentiment repository.

Implements ArticleSentimentRepository port.
Persists per-article sentiment analysis results to PostgreSQL.
"""

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.domain.trading.entities import ArticleSentiment
from app.domain.trading.ports import ArticleSentimentRepository

logger = logging.getLogger(__name__)


class ArticleSentimentRepositoryAdapter(ArticleSentimentRepository):
    """Persists article sentiment results to PostgreSQL.

    Implements the ArticleSentimentRepository port defined in the domain layer.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, result: ArticleSentiment) -> None:
        """Persist a single article sentiment result.

        Uses INSERT ON CONFLICT to skip duplicates gracefully.

        Args:
            result: An ArticleSentiment entity to persist.
        """
        query = text(
            """
            INSERT INTO article_sentiments
                (article_id, sentiment_label, sentiment_score, confidence)
            VALUES
                (:article_id, :label, :score, :confidence)
            ON CONFLICT (article_id) DO NOTHING
            """
        )

        with self._engine.begin() as conn:
            conn.execute(
                query,
                {
                    "article_id": result.article_id,
                    "label": result.sentiment_label,
                    "score": result.sentiment_score,
                    "confidence": float(result.confidence) if result.confidence else None,
                },
            )

        logger.debug("Saved sentiment for article_id=%d.", result.article_id)

    def save_batch(self, results: list[ArticleSentiment]) -> None:
        """Persist multiple article sentiment results in a single transaction.

        Uses INSERT ON CONFLICT to skip duplicates gracefully.

        Args:
            results: List of ArticleSentiment entities to persist.
        """
        if not results:
            return

        query = text(
            """
            INSERT INTO article_sentiments
                (article_id, sentiment_label, sentiment_score, confidence)
            VALUES
                (:article_id, :label, :score, :confidence)
            ON CONFLICT (article_id) DO NOTHING
            """
        )

        params = [
            {
                "article_id": r.article_id,
                "label": r.sentiment_label,
                "score": r.sentiment_score,
                "confidence": float(r.confidence) if r.confidence else None,
            }
            for r in results
        ]

        with self._engine.begin() as conn:
            conn.execute(query, params)

        logger.info("Batch-saved %d sentiment results.", len(results))
