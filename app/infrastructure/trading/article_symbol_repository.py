"""
Adapter: Article ↔ Symbol linkage repository.

Implements ArticleSymbolRepository port.
Reads/writes the article_symbols junction table.
"""

import logging
from datetime import date
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.domain.trading.ports import ArticleSymbolRepository

logger = logging.getLogger(__name__)


class ArticleSymbolRepositoryAdapter(ArticleSymbolRepository):
    """PostgreSQL adapter for article_symbols junction table."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(
        self,
        article_id: int,
        symbol: str,
        match_method: str = "keyword",
        confidence: float = 1.0,
    ) -> None:
        """Persist a single article-symbol link (upsert)."""
        query = text(
            """
            INSERT INTO article_symbols (article_id, symbol, match_method, confidence)
            VALUES (:article_id, :symbol, :method, :confidence)
            ON CONFLICT (article_id, symbol) DO NOTHING
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                query,
                {
                    "article_id": article_id,
                    "symbol": symbol,
                    "method": match_method,
                    "confidence": confidence,
                },
            )

    def save_batch(self, links: list[tuple[int, str, str, float]]) -> int:
        """Persist multiple article-symbol links.

        Args:
            links: List of (article_id, symbol, match_method, confidence).

        Returns:
            Number of rows passed (duplicates silently skipped).
        """
        if not links:
            return 0

        query = text(
            """
            INSERT INTO article_symbols (article_id, symbol, match_method, confidence)
            VALUES (:article_id, :symbol, :method, :confidence)
            ON CONFLICT (article_id, symbol) DO NOTHING
            """
        )
        params = [
            {
                "article_id": aid,
                "symbol": sym,
                "method": method,
                "confidence": conf,
            }
            for aid, sym, method, conf in links
        ]

        with self._engine.begin() as conn:
            conn.execute(query, params)

        logger.info("Batch-saved %d article-symbol links.", len(links))
        return len(links)

    def get_symbols_for_article(self, article_id: int) -> list[str]:
        """Return all symbols linked to a specific article."""
        query = text(
            "SELECT symbol FROM article_symbols WHERE article_id = :aid ORDER BY symbol"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(query, {"aid": article_id}).fetchall()
        return [r[0] for r in rows]

    def get_articles_for_symbol(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[int]:
        """Return article IDs linked to a symbol, optionally filtered by date range."""
        if start_date and end_date:
            query = text(
                """
                SELECT asym.article_id
                FROM article_symbols asym
                JOIN scraped_articles sa ON sa.id = asym.article_id
                WHERE asym.symbol = :symbol
                  AND sa.published_at::date >= :start_date
                  AND sa.published_at::date <= :end_date
                ORDER BY sa.published_at DESC
                """
            )
            params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        else:
            query = text(
                """
                SELECT article_id
                FROM article_symbols
                WHERE symbol = :symbol
                ORDER BY article_id DESC
                """
            )
            params = {"symbol": symbol}

        with self._engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r[0] for r in rows]

    def get_unlinked_article_ids(self, limit: int = 500) -> list[int]:
        """Return article IDs that have no symbol linkage yet."""
        query = text(
            """
            SELECT sa.id
            FROM scraped_articles sa
            LEFT JOIN article_symbols asym ON sa.id = asym.article_id
            WHERE asym.id IS NULL
            ORDER BY sa.published_at DESC NULLS LAST
            LIMIT :limit
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).fetchall()
        return [r[0] for r in rows]
