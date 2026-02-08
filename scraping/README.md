# Scraping — FixTrade

Quick notes for running the scraping components and the Postgres DB used by the pipeline.

1. Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

2. Bring up Postgres using Docker Compose (the DSN in `.env` uses `db` as the host):

```bash
docker compose up -d
```

3. Install dependencies and run a spider:

```bash
pip install -r requirements.txt
scrapy crawl millim
```

Notes:
- The pipeline will create its table automatically on first run.
- On Windows, `psycopg2-binary` may fail to build; this project defaults to `pg8000` (pure-Python driver) to avoid native build issues. If you want `psycopg2` instead, install the PostgreSQL client dev libraries or use Docker/WSL.
- You can also point `SCRAPING_POSTGRES_DSN` to a remote Postgres instance.
- If you prefer a GUI, connect with your favorite Postgres client to `localhost:5432`.
