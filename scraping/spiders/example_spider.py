import scrapy
from scraping.items import ExampleItem

class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        item = ExampleItem()
        item["title"] = response.css("h1::text").get()
        item["link"] = response.url
        yield item
