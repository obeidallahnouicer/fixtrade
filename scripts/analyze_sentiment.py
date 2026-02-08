#!/usr/bin/env python3
"""
CLI tool: Analyze sentiment of scraped articles.

Reads unanalyzed articles from the database, runs NLP inference,
and saves results to the article_sentiments table.

Usage:
    python scripts/analyze_sentiment.py [--batch-size=50]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine

from app.application.trading.analyze_article_sentiment import (
    AnalyzeArticleSentimentUseCase,
)
from app.application.trading.dtos import AnalyzeArticleSentimentCommand
from app.infrastructure.trading.article_sentiment_repository import (
    ArticleSentimentRepositoryAdapter,
)
from app.infrastructure.trading.scraped_article_repository import (
    ScrapedArticleRepositoryAdapter,
)
from app.infrastructure.trading.sentiment_analysis_adapter import (
    SentimentAnalysisAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_scraping_postgres_dsn() -> str:
    """Get PostgreSQL DSN from environment or use defaults."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "fixtrade")
    password = os.getenv("POSTGRES_PASSWORD", "fixtrade")
    db = os.getenv("POSTGRES_DB", "fixtrade")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def main():
    """Run sentiment analysis on unanalyzed scraped articles."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of scraped articles"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of articles to process (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting sentiment analysis CLI tool")
    logger.info("Batch size: %d", args.batch_size)

    dsn = get_scraping_postgres_dsn()
    engine = create_engine(dsn, pool_pre_ping=True)

    logger.info("Database DSN: %s", dsn)

    article_repo = ScrapedArticleRepositoryAdapter(engine)
    sentiment_repo = ArticleSentimentRepositoryAdapter(engine)
    sentiment_port = SentimentAnalysisAdapter(engine=engine)

    use_case = AnalyzeArticleSentimentUseCase(
        article_repo=article_repo,
        sentiment_repo=sentiment_repo,
        sentiment_port=sentiment_port,
    )

    command = AnalyzeArticleSentimentCommand(batch_size=args.batch_size)

    try:
        result = use_case.execute(command)
        logger.info("=" * 60)
        logger.info("SENTIMENT ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info("Total analyzed:   %d", result.total_analyzed)
        logger.info("Positive:         %d", result.positive_count)
        logger.info("Negative:         %d", result.negative_count)
        logger.info("Neutral:          %d", result.neutral_count)
        logger.info("Failed:           %d", result.failed_count)
        logger.info("=" * 60)
        return 0
    except Exception:
        logger.exception("Sentiment analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
