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
from app.core.config import settings as app_settings

SCRAPING_POSTGRES_DSN = os.getenv("SCRAPING_POSTGRES_DSN") or app_settings.get_scraping_postgres_dsn()

# Enable the Postgres pipeline
ITEM_PIPELINES = {
    "scraping.pipelines.PostgresPipeline": 300,
}

