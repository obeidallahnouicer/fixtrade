import os

BOT_NAME = "scraping"

SPIDER_MODULES = ["scraping.spiders"]
NEWSPIDER_MODULE = "scraping.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Polite defaults (adjust as needed)
DOWNLOAD_DELAY = 1.0
CONCURRENT_REQUESTS = 8
USER_AGENT = os.getenv("SCRAPER_USER_AGENT", "fixtrade-scraper/1.0 (+https://example.com)")

# Postgres DSN (set via env var SCRAPING_POSTGRES_DSN)
# Example: postgresql://user:password@localhost:5432/scraping_db
# Falls back to building DSN from individual POSTGRES_* env vars.

def _build_scraping_postgres_dsn() -> str:
    """Build the Postgres DSN from environment variables.

    Priority:
    1. Explicit SCRAPING_POSTGRES_DSN env var.
    2. Constructed from POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST,
       POSTGRES_PORT, POSTGRES_DB.
    """
    explicit_dsn = os.getenv("SCRAPING_POSTGRES_DSN")
    if explicit_dsn:
        return explicit_dsn

    user = os.getenv("POSTGRES_USER", "fixtrade")
    password = os.getenv("POSTGRES_PASSWORD", "fixtrade")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "fixtrade")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


SCRAPING_POSTGRES_DSN = _build_scraping_postgres_dsn()

# Enable the Postgres pipeline
ITEM_PIPELINES = {
    "scraping.pipelines.PostgresPipeline": 300,
}

