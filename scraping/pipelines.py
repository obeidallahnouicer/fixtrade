import logging
import json
import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, UniqueConstraint
from sqlalchemy.exc import IntegrityError, OperationalError
from dateutil import parser as dateparser
import scrapy

logger = logging.getLogger(__name__)

class PostgresPipeline:
    """Pipeline that stores scraped ArticleItem into a Postgres table `scraped_articles`.

    If DB is unreachable, the pipeline will disable DB writes and write to a local
    fallback JSONL file (configurable via SCRAPING_FALLBACK_PATH) so spiders can
    continue to run during development.
    """

    def __init__(self, dsn, fallback_path=None):
        self.dsn = dsn
        self.engine = None
        self.metadata = None
        self.articles = None
        self.enabled = True
        self.fallback_path = fallback_path or os.getenv("SCRAPING_FALLBACK_PATH", "scraped_fallback.jsonl")

    @classmethod
    def from_crawler(cls, crawler):
        dsn = crawler.settings.get("SCRAPING_POSTGRES_DSN")
        fallback = os.getenv("SCRAPING_FALLBACK_PATH") or crawler.settings.get("SCRAPING_FALLBACK_PATH")
        return cls(dsn, fallback_path=fallback)

    def open_spider(self, spider):
        if not self.dsn:
            spider.logger.warning("SCRAPING_POSTGRES_DSN is not set. PostgresPipeline will be disabled and fallback file will be used: %s", self.fallback_path)
            self.enabled = False
            return

        try:
            self.engine = create_engine(self.dsn, pool_pre_ping=True)
            self.metadata = MetaData()
            self.articles = Table(
                "scraped_articles",
                self.metadata,
                Column("id", Integer, primary_key=True),
                Column("url", String(1024), nullable=False),
                Column("title", String(512)),
                Column("summary", Text),
                Column("content", Text),
                Column("published_at", DateTime),
                UniqueConstraint("url", name="uix_url"),
            )
            # Try to create tables; if DB not reachable, move to fallback mode
            self.metadata.create_all(self.engine)
            spider.logger.info("PostgresPipeline connected to DB")
            # Log connection info and verify table presence
            try:
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    spider.logger.info("DB engine url: %s", str(self.engine.url))
                    try:
                        cnt = conn.execute(text("SELECT count(*) FROM scraped_articles")).scalar()
                        spider.logger.info("scraped_articles rows at open: %s", cnt)
                    except Exception as e:
                        spider.logger.debug("scraped_articles table not present yet: %s", e)
            except Exception:
                spider.logger.debug("Could not run verification query on DB")
        except (OperationalError, Exception) as e:
            spider.logger.warning("Could not connect to Postgres (%s). Falling back to JSONL file '%s'.", e, self.fallback_path)
            self.enabled = False

    def _write_fallback(self, item):
        try:
            with open(self.fallback_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(item), default=str, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to write fallback item to %s", self.fallback_path)

    def process_item(self, item, spider):
        # If pipeline is disabled or engine not ready, write to fallback file
        if not self.enabled or not self.engine:
            spider.logger.info("DB unavailable — writing item to fallback file: %s", self.fallback_path)
            self._write_fallback(item)
            return item

        # Normalize date (accept str or datetime)
        published_at = None
        try:
            dval = item.get("date")
            if dval:
                # if already a datetime, use directly
                import datetime
                if isinstance(dval, datetime.datetime):
                    published_at = dval
                else:
                    published_at = dateparser.parse(dval)
        except Exception:
            spider.logger.debug("Could not parse date: %s", item.get("date"))

        ins = self.articles.insert().values(
            url=item.get("url"),
            title=item.get("title"),
            summary=item.get("summary"),
            content=item.get("content"),
            published_at=published_at,
        )
        # Use a transaction context so inserts are committed immediately
        try:
            from sqlalchemy import text
            with self.engine.begin() as conn:
                conn.execute(ins)
                # quick verification query to make sure rows are visible
                try:
                    cnt = conn.execute(text("SELECT count(*) FROM scraped_articles")).scalar()
                    spider.logger.info("Inserted: %s (total rows now: %s)", item.get("url"), cnt)
                except Exception as e:
                    spider.logger.debug("Insert verification failed: %s", e)
            return item
        except IntegrityError:
            spider.logger.debug("Duplicate url skipped: %s", item.get("url"))
            raise scrapy.exceptions.DropItem("Duplicate item %s" % item.get("url"))
        except Exception as e:
            spider.logger.error("DB insert error: %s — falling back to JSONL", e)
            # fallback to JSON file to avoid losing items
            self._write_fallback(item)
            return item
