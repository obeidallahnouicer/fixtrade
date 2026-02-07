import scrapy
from scraping.items import ArticleItem

class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        item = ArticleItem()
        item["title"] = response.css("h1::text").get()
        item["url"] = response.url
        item["summary"] = ""
        item["content"] = ""
        yield item
