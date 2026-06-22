FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-scraper.txt ./
RUN pip install --no-cache-dir -r requirements-scraper.txt

COPY . /app

RUN useradd -m scrapuser && chown -R scrapuser:scrapuser /app
USER scrapuser

CMD ["scrapy", "crawl", "millim"]
