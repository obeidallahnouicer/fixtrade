# FixTrade — Minimal FastAPI Template (WSGI)

A small, clean, modular FastAPI template with a WSGI entrypoint.

Quick start

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run locally (ASGI / dev):

```bash
python -m uvicorn app.main:app --reload
```

- Run as WSGI (production example with gunicorn):

```bash
pip install gunicorn
gunicorn app.wsgi:application -w 4
```

- Or with waitress:

```bash
pip install waitress
waitress-serve --port=8080 app.wsgi:application
```

Run tests:

```bash
pytest -q
```

---

## Development: Scraping DB (Postgres via Docker) 🔧

Use Docker Compose to run a local Postgres instance used by the scraping pipeline.

1. Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

2. Start Postgres:

```bash
docker compose up -d
```

3. Install dependencies and run a spider (example):

```bash
pip install -r requirements.txt
scrapy crawl millim
```

If you prefer to run the scraper inside Docker (recommended for reproducible runs), build and run the `scraper` service which runs inside the same network as the database:

```bash
docker compose up -d --build db
# run the scraper once
docker compose run --rm scraper
```

The scraping pipeline will use `SCRAPING_POSTGRES_DSN` (from `.env`) and will create the `scraped_articles` table automatically on first run. If the DB is unreachable the pipeline will append items to `SCRAPING_FALLBACK_PATH` (default: `scraped_fallback.jsonl`).

