import scrapy
from scraping.items import ArticleItem

class MillimSpider(scrapy.Spider):
    name = "millim"
    allowed_domains = ["millim.tn"]
    start_urls = ["https://www.millim.tn/"]

    def parse(self, response):
        # follow obvious article links
        for href in response.css('a[href*="/article/"]::attr(href)').getall():
            yield response.follow(href, self.parse_article)

        # follow pagination if present
        for href in response.css('a[rel="next"]::attr(href)').getall():
            yield response.follow(href, self.parse)

    def parse_article(self, response):
        from scraping.utils import parse_date_text

        item = ArticleItem()
        item["url"] = response.url
        item["title"] = (response.css("h1::text").get() or "").strip()

        # Try several places to find a date: time[datetime], meta tags, or textual nodes
        date_text = (
            response.css('time::attr(datetime)').get()
            or response.css('meta[property="article:published_time"]::attr(content)').get()
            or response.css('.post-meta::text').get()
            or response.css('.entry-meta::text').get()
            or " ".join(response.xpath('//*[contains(@class, "date") or contains(@class, "posted")]/text()').getall())
        )
        # Fallback: sometimes the page shows a "PUBLISHED ON ..." label in the body
        if not date_text or not date_text.strip():
            import re
            m = re.search(r'published on\s*([^\n!•<]+)', response.text, flags=re.IGNORECASE)
            if m:
                date_text = m.group(1).strip()
            else:
                # French variant (PUBLIÉ LE / PUBLIÉ):
                m2 = re.search(r'publi[eé]\s*(?:le)?\s*([^\n!•<]+)', response.text, flags=re.IGNORECASE)
                if m2:
                    date_text = m2.group(1).strip()

        item["date"] = parse_date_text(date_text)

        # summary: first paragraph
        item["summary"] = (response.css('article p::text').get() or "").strip()
        # content: join paragraphs inside article
        paragraphs = response.css('article p::text').getall()
        if not paragraphs:
            # fallback to main content div
            paragraphs = response.css('.entry-content p::text').getall()
        item["content"] = "\n".join([p.strip() for p in paragraphs if p.strip()])

        yield item
