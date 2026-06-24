# 05 — Backend Architecture

## Overview

The FixTrade backend is a **FastAPI modular monolith** organized into four layers following hexagonal architecture principles. The composition root is `app/main.py`, which wires routers, middleware, error handlers, and database initialization.

---

## Layer Structure

```
app/
├── main.py                 # Composition root
├── core/
│   ├── config.py           # Pydantic Settings (.env)
│   └── db.py               # SQLAlchemy engine, create_tables()
├── domain/                 # Pure business logic (no framework imports)
│   ├── auth/
│   └── trading/
├── application/            # Use case orchestration
│   ├── auth/
│   └── trading/
├── infrastructure/         # Adapters (PostgreSQL, ML, NLP)
│   ├── auth/
│   └── trading/
├── interfaces/             # HTTP routers and Pydantic schemas
│   ├── auth/
│   ├── dashboard/
│   ├── trading/
│   └── health.py
├── ai/                     # AI agent (portfolio, MPT, LLM) — parallel module
├── nlp/                    # Sentiment analyzer
├── ml_service/             # Standalone ML FastAPI app
└── shared/
    ├── errors/handlers.py
    ├── logging.py
    └── security/
```

---

## API Reference

**Base URL:** `http://localhost:8000/api/v1`

All routes mounted in `app/main.py` lines 139–144.

### Health

| Method | Route | Auth | Request | Response | File |
|--------|-------|------|---------|----------|------|
| GET | `/health` | No | — | `{ status, version }` | `interfaces/health.py` |

### Authentication

| Method | Route | Auth | Request Body | Response | Rate Limit |
|--------|-------|------|--------------|----------|------------|
| POST | `/auth/register` | No | `{ email, password, full_name? }` | `{ access_token, token_type, user }` | 5/min |
| POST | `/auth/login` | No | `{ email, password }` | `{ access_token, token_type, user }` | 10/min |
| GET | `/auth/me` | **Bearer JWT** | — | `{ id, email, role, created_at }` | default |

**Schemas:** `app/interfaces/auth/schemas.py`  
**Router:** `app/interfaces/auth/router.py`

### Dashboard (BFF)

| Method | Route | Auth | Query | Response |
|--------|-------|------|-------|----------|
| GET | `/dashboard/bootstrap` | No | `symbol` (required) | `DashboardBootstrapResponse` |

**Response fields:** `symbol`, `historical_prices[]`, `predictions[]`, `sentiment`, `anomalies[]`, `recommendation`, `warnings[]`

**Router:** `app/interfaces/dashboard/router.py`

### Trading

| Method | Route | Auth | Request | Response |
|--------|-------|------|---------|----------|
| POST | `/trading/predictions` | No | `{ symbol, horizon_days: 1-5 }` | `{ predictions[] }` |
| POST | `/trading/sentiment` | No | `{ symbol, target_date? }` | `{ symbol, date, score, sentiment, article_count }` |
| POST | `/trading/anomalies` | No | `{ symbol }` | `{ anomalies[] }` |
| GET | `/trading/anomalies/recent` | No | `symbol?`, `limit?`, `hours_back?` | `{ anomalies[] }` |
| POST | `/trading/recommendations` | No | `{ symbol, portfolio_id? }` | `{ symbol, action, confidence, reasoning }` |
| POST | `/trading/predictions/volume` | No | `{ symbol, horizon_days }` | `{ predictions[] }` |
| POST | `/trading/predictions/liquidity` | No | `{ symbol, horizon_days }` | `{ forecasts[] }` |
| POST | `/trading/sentiment/analyze` | No | `{ article_ids? }` | `{ analyzed_count, results[] }` |
| POST | `/trading/sentiment/aggregate` | No | `{ symbol, date? }` | `{ symbol, date, score, sentiment, article_count }` |
| POST | `/trading/sentiment/link-symbols` | No | `{ article_ids? }` | `{ linked_count }` |
| POST | `/trading/anomalies/evaluate` | No | `{ symbol, start_date?, end_date? }` | `{ precision, recall, f1, ... }` |
| POST | `/trading/anomalies/intraday` | No | `{ symbol, date? }` | `{ anomalies[] }` |

**Schemas:** `app/interfaces/trading/schemas.py` (483 lines)  
**Router:** `app/interfaces/trading/router.py`  
**DI:** `app/interfaces/trading/dependencies.py`

### AI Agent — prefix `/ai`

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/ai/profile/questionnaire` | Recommend risk profile from questionnaire |
| POST | `/ai/portfolio/create` | Create virtual portfolio |
| GET | `/ai/portfolio/{id}/snapshot` | Current holdings and valuation |
| GET | `/ai/portfolio/{id}/performance` | ROI, Sharpe, drawdown metrics |
| GET | `/ai/portfolio/{id}/performance/explain` | LLM performance narrative |
| GET | `/ai/portfolio/{id}/position/{symbol}` | Single position detail |
| GET | `/ai/recommendations` | Daily ranked recommendations |
| GET | `/ai/recommendations/{symbol}/explain` | LLM trade explanation |
| POST | `/ai/portfolio/{id}/trade` | Execute buy/sell |
| POST | `/ai/portfolio/{id}/prices/update` | Refresh position mark-to-market |
| POST | `/ai/portfolio/{id}/stop-loss/check` | Evaluate stop-loss triggers |
| GET | `/ai/status` | AI module health and config |

**Router:** `app/ai/router.py`

### Portfolio Optimization — prefix `/ai/portfolio`

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/ai/portfolio/optimize` | MPT minimum variance / max Sharpe |
| POST | `/ai/portfolio/efficient-frontier` | Efficient frontier points |
| POST | `/ai/portfolio/recommendations/detailed` | Ranked signals with metrics |
| POST | `/ai/portfolio/simulate` | Backtest simulation |
| POST | `/ai/portfolio/explain` | LLM portfolio narrative |

**Router:** `app/ai/router_extended.py`

### ML Service (Standalone — port 8001)

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predictions` | Price predictions |
| POST | `/api/v1/predictions/volume` | Volume predictions |
| POST | `/api/v1/predictions/liquidity` | Liquidity classification |

**Entry:** `app/ml_service/main.py`

---

## Service Layer (Use Cases)

Each use case is a single class with one `execute()` method — application service pattern.

| Use Case | File | Port Dependencies |
|----------|------|-------------------|
| `PredictPriceUseCase` | `application/trading/predict_price.py` | `PricePredictionPort`, `StockPriceRepository` |
| `PredictVolumeUseCase` | `application/trading/predict_volume.py` | `VolumePredictionPort` |
| `PredictLiquidityUseCase` | `application/trading/predict_liquidity.py` | `LiquidityPredictionPort` |
| `GetSentimentUseCase` | `application/trading/get_sentiment.py` | `SentimentScoreRepository` |
| `AnalyzeArticleSentimentUseCase` | `application/trading/analyze_article_sentiment.py` | `SentimentAnalysisPort`, article repos |
| `LinkArticleSymbolsUseCase` | `application/trading/link_article_symbols.py` | `ArticleSymbolRepository`, matcher |
| `AggregateDailySentimentUseCase` | `application/trading/aggregate_daily_sentiment.py` | Sentiment repos |
| `DetectAnomaliesUseCase` | `application/trading/detect_anomalies.py` | `AnomalyDetectionPort` |
| `DetectIntradayAnomaliesUseCase` | `application/trading/detect_intraday_anomalies.py` | Intraday repos |
| `EvaluateAnomaliesUseCase` | `application/trading/evaluate_anomalies.py` | `KnownAnomalyRepository` |
| `GetRecentAnomaliesUseCase` | `application/trading/get_recent_anomalies.py` | `AnomalyAlertRepository` |
| `GetRecommendationUseCase` | `application/trading/get_recommendation.py` | `DecisionEnginePort` |
| `RegisterUserUseCase` | `application/auth/register_user.py` | `UserRepository`, hasher, token service |
| `LoginUserUseCase` | `application/auth/login_user.py` | Same |

### Dependency Injection Wiring

`app/interfaces/trading/dependencies.py` constructs adapter instances and injects them into use cases via FastAPI `Depends()`:

```python
# Conceptual pattern from dependencies.py
def get_predict_price_use_case() -> PredictPriceUseCase:
    return PredictPriceUseCase(
        prediction_port=PricePredictionAdapter(),
        price_repo=StockPriceRepositoryAdapter(),
    )
```

---

## Domain Layer

### Entities — `app/domain/trading/entities.py`

| Entity | Key Fields |
|--------|------------|
| `StockPrice` | symbol, date, open, close, high, low, volume |
| `PricePrediction` | symbol, target_date, predicted_close, confidence bounds |
| `VolumePrediction` | symbol, target_date, predicted_volume |
| `LiquidityForecast` | symbol, tier probabilities |
| `SentimentScore` | symbol, date, score (-1 to 1), sentiment label |
| `AnomalyAlert` | id, symbol, anomaly_type, severity, description |
| `Portfolio` | id, risk_profile, cash_balance, positions |
| `TradeRecommendation` | symbol, action, confidence, reasoning |

### Port Interfaces — `app/domain/trading/ports.py`

12 abstract base classes including `StockPriceRepository`, `PricePredictionPort`, `AnomalyDetectionPort`, `SentimentAnalysisPort`, `DecisionEnginePort`, `PortfolioRepository`.

---

## Asynchronous Processing

### Current State

| Mechanism | Status | Location |
|-----------|--------|----------|
| FastAPI async endpoints | Available | Routers use `async def` where I/O-bound |
| APScheduler retraining | Implemented, disabled at startup | `prediction/realtime/scheduler.py` |
| Scrapy crawler worker | Active in Docker | `scripts/scraper_worker.py` |
| WebSocket/SSE streaming | Disabled | `app/main.py` lifespan (commented) |
| Background tasks | Not used | No Celery/RQ |

### Scraper Worker Loop

```python
# scripts/scraper_worker.py — conceptual
while True:
    subprocess.run(["scrapy", "crawl", "millim"])
    write_timestamp("data/last_scrape.txt")
    sleep(SCRAPER_INTERVAL_SECONDS)  # default 3600
```

### Why Async Processing?

| Use Case | Rationale |
|----------|-----------|
| Scraping | Long-running, I/O-bound; shouldn't block API |
| Model retraining | CPU-intensive; scheduled off-peak via APScheduler |
| Real-time streaming | Future: push predictions without polling |

---

## External Integrations

| Integration | Adapter / Module | Protocol |
|-------------|------------------|----------|
| PostgreSQL | `*_repository.py` adapters | SQL via SQLAlchemy / raw SQL |
| Redis | `prediction/utils/cache.py` | Redis protocol |
| ML inference | `PricePredictionAdapter` | Local import or HTTP to ml-service |
| HuggingFace | `app/nlp/sentiment.py` | Local model inference |
| Groq / LiteLLM | `app/ai/llm_explainer.py` | HTTPS REST |
| BVMT CSV files | `prediction/etl/extract/bvmt_extractor.py` | Filesystem |
| News websites | `scraping/spiders/` | HTTP (Scrapy) |

---

## Error Handling

Centralized mapping in `app/shared/errors/handlers.py`:

| Domain Exception | HTTP Status |
|------------------|-------------|
| `SymbolNotFoundError` | 404 |
| `InvalidHorizonError` | 422 |
| `InsufficientDataError` | 400 |
| `InvalidCredentialsError` | 401 |
| `UserAlreadyExistsError` | 400 |
| `AnomalyDetectionFailedError` | 500 |
| Generic `Exception` | 500 (no stack trace exposed) |

---

## Middleware Stack

Applied in order ( `app/main.py`):

1. **CORSMiddleware** — localhost:3000, localhost:8501
2. **slowapi rate limiter** — default 60/min (`RATE_LIMIT_DEFAULT`)
3. **SecurityHeadersMiddleware** — CSP, X-Frame-Options, nosniff
4. **Domain error handlers** — registered via `register_error_handlers(app)`

---

## MVP Limitations (Verified in Code)

| Component | Limitation | File |
|-----------|------------|------|
| `DecisionEngineAdapter` | Always returns BUY, confidence 0.85 | `infrastructure/trading/decision_engine_adapter.py` |
| `PortfolioRepositoryAdapter` | Returns empty portfolio without DB lookup | `infrastructure/trading/portfolio_repository.py` |
| AI recommendations | In-memory agents; some TODOs for DB | `app/ai/router.py` |
| Trading routes | No JWT requirement | All trading routers |

---

## Related Documentation

- [06-authentication-security.md](06-authentication-security.md)
- [07-database-design.md](07-database-design.md)
- [14-design-patterns.md](14-design-patterns.md)
