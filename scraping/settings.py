BOT_NAME = "scraping"

SPIDER_MODULES = ["scraping.spiders"]
NEWSPIDER_MODULE = "scraping.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure item pipelines
# ITEM_PIPELINES = {
#    "scraping.pipelines.ScrapingPipeline": 300,
# }
