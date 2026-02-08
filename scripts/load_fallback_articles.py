#!/usr/bin/env python3
"""
Load scraped articles from JSONL fallback file into database.

Usage:
    python scripts/load_fallback_articles.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dateutil import parser as dateparser
from sqlalchemy import create_engine, text

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
    """Load articles from fallback JSONL file into PostgreSQL."""
    logger.info("Starting fallback article loader")

    fallback_path = Path("scraped_fallback.jsonl")
    if not fallback_path.exists():
        logger.error("Fallback file not found: %s", fallback_path)
        return 1

    dsn = get_scraping_postgres_dsn()
    engine = create_engine(dsn, pool_pre_ping=True)

    logger.info("Database DSN: %s", dsn)
    logger.info("Loading articles from: %s", fallback_path)

    inserted = 0
    skipped = 0

    with open(fallback_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                published_at = None
                if item.get("date"):
                    try:
                        if isinstance(item["date"], str):
                            published_at = dateparser.parse(item["date"])
                        elif isinstance(item["date"], datetime):
                            published_at = item["date"]
                    except Exception:
                        pass

                query = text(
                    """
                    INSERT INTO scraped_articles
                        (url, title, summary, content, published_at)
                    VALUES
                        (:url, :title, :summary, :content, :published_at)
                    ON CONFLICT (url) DO NOTHING
                    RETURNING id
                    """
                )

                with engine.begin() as conn:
                    result = conn.execute(
                        query,
                        {
                            "url": item.get("url"),
                            "title": item.get("title"),
                            "summary": item.get("summary"),
                            "content": item.get("content"),
                            "published_at": published_at,
                        },
                    )
                    if result.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1

            except Exception:
                logger.exception("Failed to insert article: %s", line[:100])
                continue

    logger.info("=" * 60)
    logger.info("LOADING COMPLETE")
    logger.info("=" * 60)
    logger.info("Inserted: %d", inserted)
    logger.info("Skipped:  %d", skipped)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
