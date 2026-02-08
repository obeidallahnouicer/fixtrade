import scrapy

class ArticleItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    summary = scrapy.Field()
    content = scrapy.Field()
