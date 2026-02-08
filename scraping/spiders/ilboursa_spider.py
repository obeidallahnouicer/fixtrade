import scrapy
from scraping.items import ArticleItem

class IlboursaSpider(scrapy.Spider):
    name = "ilboursa"
    allowed_domains = ["ilboursa.com"]
    start_urls = ["https://www.ilboursa.com/"]

    def parse(self, response):
        # article links typically contain /marches/ or /magazine/ etc.
        for href in response.css('a[href*="/marches/"]::attr(href), a[href*="/magazine/"]::attr(href)').getall():
            yield response.follow(href, self.parse_article)

        # pagination
        for href in response.css('a[rel="next"]::attr(href)').getall():
            yield response.follow(href, self.parse)

    def parse_article(self, response):
        from scraping.utils import parse_date_text

        item = ArticleItem()
        item["url"] = response.url
        item["title"] = (response.css("h1::text").get() or "").strip()

        # Try several selectors and text nodes to find a date (support French & English)
        date_text = (
            response.css('time::attr(datetime)').get()
            or response.css('meta[property="article:published_time"]::attr(content)').get()
            or response.css('.date::text').get()
            or response.xpath('//*[contains(@class, "date")]/text()').get()
            or " ".join(response.xpath('//*[contains(@class, "date") or contains(@class, "posted")]/text()').getall())
        )
        # Fallbacks: look for common English/French labels inside the page when selectors miss it
        if not date_text or not date_text.strip():
            import re
            m = re.search(r'published on\s*([^\n!•<]+)', response.text, flags=re.IGNORECASE)
            if m:
                date_text = m.group(1).strip()
            else:
                m2 = re.search(r'publi[eé]\s*(?:le)?\s*([^\n!•<]+)', response.text, flags=re.IGNORECASE)
                if m2:
                    date_text = m2.group(1).strip()

        item["date"] = parse_date_text(date_text)

        item["summary"] = (response.css('.lead::text').get() or response.css('.intro::text').get() or "").strip()
        paragraphs = response.css('.article-content p::text').getall() or response.css('article p::text').getall()
        item["content"] = "\n".join([p.strip() for p in paragraphs if p.strip()])
        yield item
