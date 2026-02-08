"""
IlBoursa Historical Spider - Scrapes articles by date range.

Usage:
    scrapy crawl ilboursa_historical -a start_date=2016-01-01 -a end_date=2025-12-31
    scrapy crawl ilboursa_historical -a start_date=2024-01-01 -a end_date=2024-12-31
"""
import scrapy
from datetime import datetime, timedelta
from scraping.items import ArticleItem
from scraping.utils import parse_date_text


class IlboursaHistoricalSpider(scrapy.Spider):
    name = "ilboursa_historical"
    allowed_domains = ["ilboursa.com"]
    
    def __init__(self, start_date="2016-01-01", end_date="2025-12-31", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Generate all dates in the range
        self.dates_to_scrape = []
        current = self.start_date
        while current <= self.end_date:
            self.dates_to_scrape.append(current)
            current += timedelta(days=1)
        
        self.logger.info(f"Will scrape {len(self.dates_to_scrape)} days from {start_date} to {end_date}")
    
    def start_requests(self):
        """Generate requests for each date in the range."""
        base_url = "https://www.ilboursa.com/marches/actualites_bourse_tunis"
        
        for date in self.dates_to_scrape:
            # Format: DD/MM/YYYY for the URL parameter
            date_str = date.strftime("%d/%m/%Y")
            url = f"{base_url}?date={date_str}"
            
            yield scrapy.Request(
                url=url,
                callback=self.parse_date_page,
                meta={"date": date},
                dont_filter=True,  # Allow multiple requests to same base URL
            )
    
    def parse_date_page(self, response):
        """Parse the list of articles for a specific date."""
        scraped_date = response.meta["date"]
        
        # Extract article links from the table
        # Structure: table with rows containing time, title link, and stock price
        article_links = response.css('table tr td:nth-child(2) a::attr(href)').getall()
        
        if article_links:
            self.logger.info(f"Found {len(article_links)} articles on {scraped_date}")
        
        for href in article_links:
            if href and '/marches/' in href:
                yield response.follow(
                    href,
                    callback=self.parse_article,
                    meta={"expected_date": scraped_date}
                )
    
    def parse_article(self, response):
        """Parse individual article page."""
        item = ArticleItem()
        item["url"] = response.url
        
        # Title - try multiple selectors
        title = (
            response.css("h1.article-title::text").get()
            or response.css("h1::text").get()
            or response.css(".article-header h1::text").get()
            or ""
        ).strip()
        item["title"] = title
        
        # Date - try multiple approaches
        date_text = (
            response.css('time::attr(datetime)').get()
            or response.css('meta[property="article:published_time"]::attr(content)').get()
            or response.css('.article-date::text').get()
            or response.css('.date::text').get()
            or response.xpath('//span[contains(@class, "date")]/text()').get()
        )
        
        # Parse the date - if it fails, use the expected date from URL
        parsed_date = parse_date_text(date_text)
        if not parsed_date and response.meta.get("expected_date"):
            # Use the date from the listing page as fallback
            parsed_date = datetime.combine(
                response.meta["expected_date"],
                datetime.min.time()
            )
        
        item["date"] = parsed_date
        
        # Summary - first paragraph or lead text
        summary = (
            response.css('.article-lead::text').get()
            or response.css('.lead::text').get()
            or response.css('.intro::text').get()
            or response.css('article p:first-of-type::text').get()
            or ""
        ).strip()
        item["summary"] = summary
        
        # Content - all paragraphs
        paragraphs = (
            response.css('.article-body p::text').getall()
            or response.css('.article-content p::text').getall()
            or response.css('article p::text').getall()
            or response.css('.content p::text').getall()
        )
        
        # Clean and join paragraphs
        clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        item["content"] = "\n".join(clean_paragraphs)
        
        # Only yield if we have meaningful content
        if item["title"] and (item["content"] or item["summary"]):
            yield item
        else:
            self.logger.warning(f"Skipping article with missing content: {response.url}")
