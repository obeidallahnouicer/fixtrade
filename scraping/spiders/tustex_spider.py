"""
Tustex Spider — 3rd Tunisian financial news source.

Scrapes articles from tustex.com, one of Tunisia's leading
financial news portals covering BVMT-listed companies.

Usage:
    scrapy crawl tustex
"""

import scrapy

from scraping.items import ArticleItem


class TustexSpider(scrapy.Spider):
    """Scraper for tustex.com — Tunisian financial news portal."""

    name = "tustex"
    allowed_domains = ["tustex.com"]
    start_urls = [
        "https://www.tustex.com/bourse-de-tunis",
        "https://www.tustex.com/economie-tunisie",
        "https://www.tustex.com/entreprises",
    ]

    def parse(self, response):
        """Parse listing pages and follow article links + pagination."""
        # Article links: various patterns on tustex
        for href in response.css(
            'a[href*="/bourse-de-tunis/"]::attr(href), '
            'a[href*="/economie-tunisie/"]::attr(href), '
            'a[href*="/entreprises/"]::attr(href), '
            'a[href*="/article/"]::attr(href)'
        ).getall():
            # Avoid re-crawling the listing root
            if href != response.url:
                yield response.follow(href, self.parse_article)

        # Pagination
        for href in response.css(
            'a[rel="next"]::attr(href), '
            'li.next a::attr(href), '
            'a.pager-next::attr(href)'
        ).getall():
            yield response.follow(href, self.parse)

    def parse_article(self, response):
        """Parse a single article page."""
        from scraping.utils import parse_date_text

        item = ArticleItem()
        item["url"] = response.url

        # Title
        item["title"] = (
            response.css("h1.article-title::text").get()
            or response.css("h1::text").get()
            or ""
        ).strip()

        # Date — try multiple selectors
        date_text = (
            response.css('time::attr(datetime)').get()
            or response.css(
                'meta[property="article:published_time"]::attr(content)'
            ).get()
            or response.css(".article-date::text").get()
            or response.css(".date::text").get()
            or response.xpath(
                '//*[contains(@class, "date")]/text()'
            ).get()
            or " ".join(
                response.xpath(
                    '//*[contains(@class, "date") or '
                    'contains(@class, "posted") or '
                    'contains(@class, "time")]/text()'
                ).getall()
            )
        )

        # Fallback: look for French date patterns in body text
        if not date_text or not date_text.strip():
            import re

            m = re.search(
                r"publi[eé]\s*(?:le)?\s*([^\n!•<]+)",
                response.text,
                flags=re.IGNORECASE,
            )
            if m:
                date_text = m.group(1).strip()

        item["date"] = parse_date_text(date_text)

        # Summary / chapeau
        item["summary"] = (
            response.css(".article-chapeau::text").get()
            or response.css(".chapeau::text").get()
            or response.css(".lead::text").get()
            or response.css(".intro::text").get()
            or response.css("article p:first-of-type::text").get()
            or ""
        ).strip()

        # Content — join all paragraphs
        paragraphs = (
            response.css(".article-body p::text").getall()
            or response.css(".article-content p::text").getall()
            or response.css("article p::text").getall()
            or response.css(".content p::text").getall()
        )
        clean = [p.strip() for p in paragraphs if p.strip()]
        item["content"] = "\n".join(clean)

        # Only yield if we have meaningful content
        if item["title"] and (item["content"] or item["summary"]):
            yield item
        else:
            self.logger.warning(
                "Skipping article with missing content: %s", response.url
            )
