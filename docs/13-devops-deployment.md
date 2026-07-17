# 13 — DevOps, Deployment and Containerization

## Overview

FixTrade deploys via **Docker Compose** with optional local development using Python virtual environments. There is **no CI/CD pipeline** configured in the repository (no `.github/workflows/`).

---

## Docker Architecture

```mermaid
graph TB
    subgraph Client
        Browser["Browser :3000"]
    end

    subgraph Docker Compose
        NGINX["frontend<br/>nginx:alpine"]
        API["api<br/>FastAPI :8000"]
        ML["ml-service<br/>:8001"]
        AUTH["auth-service<br/>:8002"]
        GENAI["genai-service<br/>:8003"]
        PG["postgres:16-alpine<br/>:5432"]
        REDIS["redis:7-alpine<br/>:6379"]
        SCRAPER["scraper<br/>worker loop"]
        ETL["etl-worker<br/>one-shot"]
    end

    Browser --> NGINX
    NGINX -->|"/api/"| API
    API --> PG
    API --> REDIS
    API --> ML
    API --> AUTH
    API --> GENAI
    ML --> PG
    ML --> REDIS
    SCRAPER --> PG
    ETL --> PG
```

---

## Dockerfiles

| File | Base Image | Purpose | CMD |
|------|------------|---------|-----|
| `docker/Dockerfile` | python:3.11-slim (multi-stage) | Main API | `uvicorn app.main:app --port 8000` |
| `docker/api.Dockerfile` | python:3.11-slim | Full-repo API variant | Same uvicorn |
| `docker/ml_service.Dockerfile` | python:3.11-slim | ML microservice | `uvicorn app.ml_service.main:app --port 8001` |
| `docker/auth.Dockerfile` | python:3.11-slim | Auth microservice | `uvicorn services.auth_service.app:app --port 8002` |
| `docker/genai.Dockerfile` | python:3.11-slim | GenAI stub | `uvicorn services.genai_service.app:app --port 8003` |
| `docker/scraper.Dockerfile` | python:3.11-slim | Scrapy container | `python scripts/scraper_worker.py` |
| `docker/worker.Dockerfile` | python:3.11-slim | ETL worker | `python run_training.py` or fallback loader |
| `docker/frontend.Dockerfile` | node:18-alpine | Vite dev (optional) | `npm run dev -- --host` |
| `Dockerfile` (root) | python:3.12-slim | Alternate scraper | `scrapy crawl millim` |

### Multi-Stage Build — `docker/Dockerfile`

| Stage | Purpose |
|-------|---------|
| **Builder** | Install Python dependencies from `requirements.txt` |
| **Production** | Copy installed packages into slim image |
| **Security** | Non-root user (`appuser`), pre-created data directories |

---

## Docker Compose Services — `docker-compose.yml`

| Service | Image/Build | Port | Depends On | Restart |
|---------|-------------|------|------------|---------|
| postgres | postgres:16-alpine | 5432 | — | default |
| redis | redis:7-alpine | 6379 | — | default |
| api | docker/Dockerfile | 8000 | postgres, redis, ml-service, auth-service, genai-service | unless-stopped |
| ml-service | docker/ml_service.Dockerfile | 8001 | postgres, redis | unless-stopped |
| auth-service | docker/auth.Dockerfile | 8002 | postgres | unless-stopped |
| genai-service | docker/genai.Dockerfile | 8003 | redis | unless-stopped |
| frontend | nginx:alpine | 3000→80 | — | unless-stopped |
| etl-worker | docker/worker.Dockerfile | — | postgres | no (one-shot) |
| scraper | docker/scraper.Dockerfile | — | postgres | unless-stopped |

### Local Infrastructure Only — `docker-compose.local.yml`

| Service | Port | Purpose |
|---------|------|---------|
| postgres | 5432 | Database |
| redis | 6379 | Cache |
| pgadmin | 5050 | DB admin UI |

---

## Networks and Volumes

### Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `pgdata` | PostgreSQL data dir | Persistent database |
| `./db` | `/docker-entrypoint-initdb.d` | Auto-run schema SQL on first start |
| `./data` | `/app/data` | Parquet data lake |
| `./models` | `/app/models` | Trained model artifacts |
| `./frontend/dist` | nginx html root | Static SPA |
| `.:/app:rw` | API/ML/scraper containers | Dev bind mount (hot reload) |

### Networks

Default Docker Compose bridge network — all services communicate by service name (e.g., `postgres`, `redis`, `ml-service`).

---

## Nginx Configuration — `docker/nginx.conf`

| Route | Target | Purpose |
|-------|--------|---------|
| `/` | Static files in `/usr/share/nginx/html` | SPA assets |
| `/api/` | `host.docker.internal:8000` | Proxy to host API |
| `/_probe` | API `/api/v1/health` | Health check |
| Fallback | `/index.html` | SPA client-side routing |

**Note:** nginx proxies to `host.docker.internal:8000` rather than the `api` compose service — assumes API runs on host or requires network adjustment for full in-compose deployment.

---

## Health Checks

| Service | Check | Interval |
|---------|-------|----------|
| postgres | `pg_isready -U fixtrade` | 5s |
| redis | `redis-cli ping` | 5s |
| api | HTTP GET `/api/v1/health` | 10s |
| ml-service | HTTP GET `/api/v1/health` | 10s |
| auth-service | HTTP GET `/api/v1/health` | 10s |
| genai-service | HTTP GET `/api/v1/health` | 10s |
| scraper | `last_scrape.txt` within 24h | 60s |

---

## Startup Scripts

| Script | Platform | Actions |
|--------|----------|---------|
| `scripts/start_services.sh` | Linux/macOS | `docker compose up`, wait for health, ETL, smoke test |
| `scripts/start_services.ps1` | Windows | Same workflow |
| `scripts/smoke_test.py` | All | Hit health on ports 8000–8003, write JSON report |
| `quickstart.sh` | Linux/macOS | Training + predictions quick start |
| `start_dev.sh` | Linux/macOS | Load `.env`, start uvicorn |
| `run_app.py` | All | Safe startup with explicit `.env` loading |

---

## Environment Configuration

### Required Variables (`.env`)

| Variable | Purpose | Docker Override |
|----------|---------|-----------------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://fixtrade:fixtrade@postgres:5432/fixtrade` |
| `REDIS_URL` | Redis connection | `redis://redis:6379/0` |
| `AUTH_SECRET_KEY` | JWT signing | `dev-secret-key-change-in-production` ⚠️ |
| `ML_SERVICE_URL` | Remote ML delegation | `http://ml-service:8001` |
| `POSTGRES_HOST` | DB hostname | `postgres` |
| `REDIS_HOST` | Cache hostname | `redis` |
| `FIXTRADE_DATA_DIR` | Data lake path | `/app/data` |
| `MODEL_DIR` | Model artifacts | `/app/models` |
| `GROQ_API_KEY` | LLM explainability | From `.env` (optional) |
| `SCRAPING_POSTGRES_DSN` | Scraper DB | From `.env` or built from POSTGRES_* |

Full variable reference in [README.md](../README.md) Configuration section.

### Secret Management

| Environment | Approach |
|-------------|----------|
| Local dev | `.env` file (gitignored) |
| Docker Compose | `env_file: .env` + environment overrides |
| Production (recommended) | Docker secrets, Vault, or cloud parameter store |

**Warning:** Default `AUTH_SECRET_KEY` and `POSTGRES_PASSWORD=fixtrade` in compose are for development only.

---

## CI/CD

**Status:** Not implemented.

| Missing Component | Recommendation |
|-------------------|----------------|
| GitHub Actions | pytest on PR, Docker build, lint |
| Automated deployment | Push to registry on merge to main |
| Integration tests in CI | docker compose up + smoke_test.py |

Test suite exists locally: `pytest` with 170+ tests across domain, application, API, prediction, and integration layers.

---

## Deployment Commands

```bash
# Full stack
docker compose up -d

# Infrastructure only
docker compose -f docker-compose.local.yml up -d

# Build and start with orchestration
./scripts/start_services.sh        # Linux/macOS
./scripts/start_services.ps1       # Windows

# Run ETL inside container
docker compose exec api python -m prediction etl

# Run training inside container
docker compose exec api python run_training.py

# View logs
docker compose logs -f api

# Stop
docker compose down
```

---

## Production Considerations

| Aspect | Current State | Production Recommendation |
|--------|---------------|--------------------------|
| HTTPS | Not configured | TLS at nginx or load balancer |
| Secret rotation | Static dev secrets | External secret manager |
| Multi-worker API | Single uvicorn worker | Gunicorn with 4+ workers |
| Database backups | Not automated | pg_dump scheduled job |
| Log aggregation | stdout only | ELK, Loki, or CloudWatch |
| Resource limits | Not set in compose | CPU/memory limits per service |
| Read-only containers | API writable via bind mount | Remove dev bind mounts |

---

## Related Documentation

- [03-technology-stack.md](03-technology-stack.md) — Dependency versions
- [06-authentication-security.md](06-authentication-security.md) — Security configuration
- [02-system-architecture.md](02-system-architecture.md) — Deployment topologies
