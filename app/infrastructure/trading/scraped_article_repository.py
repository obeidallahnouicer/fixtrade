"""
Adapter: Scraped article repository.

Implements ScrapedArticleRepository port.
Reads unanalyzed articles from the scraped_articles table.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.domain.trading.entities import ScrapedArticle
from app.domain.trading.ports import ScrapedArticleRepository

logger = logging.getLogger(__name__)


class ScrapedArticleRepositoryAdapter(ScrapedArticleRepository):
    """Reads scraped articles from PostgreSQL.

    Implements the ScrapedArticleRepository port defined in the domain layer.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_unanalyzed_articles(self, limit: int = 100) -> list[ScrapedArticle]:
        """Return articles that have no entry in article_sentiments yet.

        Args:
            limit: Maximum number of articles to return.

        Returns:
            List of ScrapedArticle entities without sentiment analysis.
        """
        query = text(
            """
            SELECT sa.id, sa.url, sa.title, sa.summary, sa.content, sa.published_at
            FROM scraped_articles sa
            LEFT JOIN article_sentiments ase ON sa.id = ase.article_id
            WHERE ase.id IS NULL
            ORDER BY sa.published_at DESC NULLS LAST
            LIMIT :limit
            """
        )

        with self._engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).fetchall()

        articles = []
        for row in rows:
            articles.append(
                ScrapedArticle(
                    id=row[0],
                    url=row[1],
                    title=row[2],
                    summary=row[3],
                    content=row[4],
                    published_at=row[5],
                )
            )

        logger.info("Fetched %d unanalyzed articles.", len(articles))
        return articles

    def get_article_text(self, article: ScrapedArticle) -> str:
        """Return the best available text for NLP analysis.

        Priority: content > summary > title.

        Args:
            article: A scraped article entity.

        Returns:
            The best available text string, or empty string.
        """
        if article.content and article.content.strip():
            return article.content
        if article.summary and article.summary.strip():
            return article.summary
        if article.title and article.title.strip():
            return article.title
        return ""
