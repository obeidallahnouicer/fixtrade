import scrapy

class ExampleItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
