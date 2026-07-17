# FixTrade

**AI-powered stock prediction and trading intelligence platform for the Tunisian stock exchange (BVMT).**

FixTrade combines ensemble ML price forecasting (LSTM, XGBoost, Prophet), multilingual NLP sentiment analysis, multi-layer market anomaly detection, and an AI portfolio agent with LLM explainability — exposed via a FastAPI REST API, React dashboard, and optional Streamlit interface.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/obeidallahnouicer/fixtrade.git
cd fixtrade
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env   # edit DATABASE_URL, REDIS_URL, etc.

# 3. Start infrastructure
docker compose -f docker-compose.local.yml up -d   # postgres + redis

# 4. Train models
python run_training.py

# 5. Start API (loads .env explicitly)
python run_app.py

# 6. Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### Docker (full stack)

```bash
docker compose up -d
# API :8000 | ML :8001 | Auth :8002 | GenAI :8003 | Frontend :3000
```

The `etl-worker` is a long-running automation service. On first startup it
loads the bundled BVMT history and article backlog, then every five minutes it
links and scores new articles, refreshes persisted forecasts, detects
anomalies, and writes recommendations. The scraper runs hourly. Model
retraining is attempted weekly; data-derived persisted forecasts remain
available if the heavyweight ML runtime is unavailable.

Check the pipeline:

```bash
curl http://localhost:8000/api/v1/dashboard/pipeline-status
docker compose logs -f etl-worker scraper
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| ML Service | http://localhost:8001/api/v1/health |
| React Dashboard | http://localhost:3000 |
| Streamlit (optional) | http://localhost:8501 |

---

## What It Does

| Capability | Description |
|------------|-------------|
| **Price Prediction** | 1–5 day closing price forecasts with confidence intervals |
| **Volume & Liquidity** | Transaction volume and liquidity tier forecasting |
| **Sentiment Analysis** | Multilingual NLP on scraped financial news |
| **Anomaly Detection** | Statistical + prediction + sentiment contradiction layers |
| **AI Portfolio Agent** | Risk profiles, MPT optimization, virtual trading, LLM explanations |
| **Web Scraping** | Automated news collection from Tunisian financial portals |

---

## Architecture at a Glance

```
React/Streamlit → FastAPI (Hexagonal Monolith) → PostgreSQL + Redis
                        ↓
              prediction/ (ETL + ML Ensemble)
                        ↓
              scraping/ (Scrapy News Crawlers)
```

- **Style:** Modular monolith with hexagonal (ports & adapters) architecture
- **Optional microservices:** ML (`:8001`), Auth (`:8002`), GenAI (`:8003`)
- **Data:** Medallion ETL — Bronze → Silver → Gold Parquet + PostgreSQL

See [docs/02-system-architecture.md](docs/02-system-architecture.md) for full diagrams.

---

## Documentation

Comprehensive technical documentation for thesis and architecture review:

| # | Document | Topic |
|---|----------|-------|
| 01 | [Project Overview](docs/01-project-overview.md) | Business problem, objectives, workflows |
| 02 | [System Architecture](docs/02-system-architecture.md) | Components, data flows, deployment topologies |
| 03 | [Technology Stack](docs/03-technology-stack.md) | Dependencies, versions, selection rationale |
| 04 | [Frontend Architecture](docs/04-frontend-architecture.md) | React, Zustand, BFF pattern |
| 05 | [Backend Architecture](docs/05-backend-architecture.md) | API reference, use cases, services |
| 06 | [Authentication & Security](docs/06-authentication-security.md) | JWT, rate limiting, security analysis |
| 07 | [Database Design](docs/07-database-design.md) | Schema, ER diagrams, indexing |
| 08 | [Data Pipeline & ETL](docs/08-data-pipeline-etl.md) | Medallion architecture, feature engineering |
| 09 | [Web Scraping](docs/09-web-scraping-system.md) | Scrapy spiders, pipelines, anti-bot |
| 10 | [Anomaly Detection](docs/10-anomaly-detection.md) | Algorithms, thresholds, integration |
| 11 | [Machine Learning](docs/11-machine-learning-prediction.md) | Models, training, inference, metrics |
| 12 | [Generative AI](docs/12-generative-ai.md) | LLM explainability, portfolio recommendations |
| 13 | [DevOps & Deployment](docs/13-devops-deployment.md) | Docker, compose, environment config |
| 14 | [Design Patterns](docs/14-design-patterns.md) | Verified patterns with code locations |
| 15 | [Development Process](docs/15-sprint-and-development-process.md) | Agile milestones from git history |

### Module-Specific Guides

- [app/ai/README.md](app/ai/README.md) — AI Decision Agent module
- [app/ai/QUICKSTART.md](app/ai/QUICKSTART.md) — AI module quick start
- [scraping/README.md](scraping/README.md) — Scraper setup

---

## Project Structure

```
fixtrade/
├── app/              # FastAPI monolith (domain, application, infrastructure, interfaces)
├── prediction/       # ML pipeline (ETL, features, models, inference)
├── scraping/         # Scrapy news crawlers
├── services/         # Standalone auth & GenAI microservices
├── frontend/         # React + Vite dashboard
├── db/               # PostgreSQL schema migrations
├── docker/           # Dockerfiles and nginx config
├── docs/             # Technical documentation (thesis source material)
├── scripts/          # Startup, scraper worker, smoke tests
└── tests/            # 170+ pytest tests
```

---

## API Overview

Base URL: `http://localhost:8000/api/v1`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/auth/register`, `/auth/login` | POST | User authentication |
| `/dashboard/bootstrap?symbol=` | GET | Aggregated dashboard data (BFF) |
| `/trading/predictions` | POST | Price forecasts |
| `/trading/sentiment` | POST | Daily sentiment score |
| `/trading/anomalies` | POST | Detect market anomalies |
| `/trading/recommendations` | POST | Buy/sell/hold signal |
| `/ai/recommendations` | GET | AI-ranked portfolio recommendations |
| `/ai/portfolio/optimize` | POST | MPT portfolio optimization |

Full API reference: [docs/05-backend-architecture.md](docs/05-backend-architecture.md)

---

## Testing

```bash
pytest                          # All tests (170+)
pytest tests/test_prediction.py # ML pipeline tests
pytest --cov=app --cov=prediction --cov-report=html
```

---

## Prerequisites

- Python 3.11+
- Node.js 18+ (frontend)
- PostgreSQL 16+ and Redis 7+ (or Docker)
- Docker & Docker Compose (recommended)

---

## License

Proprietary. All rights reserved.
