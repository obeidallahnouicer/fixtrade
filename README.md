# FixTrade

**AI-powered stock prediction and trading recommendation platform for the Tunisian stock exchange (BVMT).**

FixTrade ingests historical OHLCV market data, scrapes financial news articles, runs multilingual NLP sentiment analysis, and produces multi-day price forecasts through an ensemble of LSTM, XGBoost, and Prophet models — all exposed via a production-hardened FastAPI REST API.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Technology Stack](#technology-stack)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Data Pipeline (ETL)](#data-pipeline-etl)
11. [Feature Engineering](#feature-engineering)
12. [ML Models & Training](#ml-models--training)
13. [Inference & Caching](#inference--caching)
14. [NLP Sentiment Analysis](#nlp-sentiment-analysis)
15. [Web Scraping](#web-scraping)
16. [Database Schema](#database-schema)
17. [Testing](#testing)
18. [Docker Deployment](#docker-deployment)
19. [CLI Reference](#cli-reference)
20. [Security](#security)
21. [Monitoring & Observability](#monitoring--observability)
22. [Contributing](#contributing)
23. [License](#license)

---

## Overview

FixTrade is a modular monolith built with **Hexagonal Architecture** (Ports & Adapters) targeting the BVMT (Bourse des Valeurs Mobilières de Tunis). The system tracks 30+ listed securities and delivers:

| Capability | Description |
|---|---|
| **Price Prediction** | 1–5 day ahead closing price forecasts with confidence intervals |
| **Sentiment Analysis** | Multilingual (French/Arabic/English) NLP on financial news |
| **Anomaly Detection** | Statistical detection of volume spikes, price swings |
| **Trade Recommendations** | Buy/sell/hold signals combining predictions, sentiment, and anomalies |
| **Portfolio Tracking** | Virtual portfolio management with P&L tracking |

### Key Metrics

- **50+ engineered features** per stock per day (technical, temporal, volume, lag/momentum)
- **3 ML models** in ensemble (LSTM, XGBoost, Prophet) with liquidity-tiered selection
- **Sub-50ms** cache-hit latency, **< 2s** cache-miss inference SLA
- **Walk-forward cross-validation** with anti-leakage guarantees
- **MLflow experiment tracking** for all training runs

---

## Architecture

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INTERFACES LAYER                              │
│  FastAPI Routers · Pydantic Schemas · Input Validation               │
│  POST /predictions · POST /sentiment · POST /anomalies              │
│  POST /recommendations · GET /health                                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                       APPLICATION LAYER                              │
│  Use Cases (one class = one use case)                                │
│  PredictPriceUseCase · GetSentimentUseCase                           │
│  DetectAnomaliesUseCase · GetRecommendationUseCase                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                         DOMAIN LAYER                                 │
│  Entities: StockPrice · PricePrediction · SentimentScore             │
│            AnomalyAlert · Portfolio · TradeRecommendation            │
│  Ports (ABCs): StockPriceRepository · PricePredictionPort            │
│                SentimentAnalysisPort · AnomalyDetectionPort          │
│                PortfolioRepository · DecisionEnginePort               │
│  Errors: SymbolNotFoundError · InvalidHorizonError · ...            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                             │
│  Adapters implementing domain ports:                                 │
│  PricePredictionAdapter → prediction.inference.PredictionService     │
│  SentimentAnalysisAdapter → app.nlp.SentimentAnalyzer                │
│  StockPriceRepositoryAdapter → PostgreSQL                            │
│  PortfolioRepositoryAdapter → PostgreSQL                             │
│  AnomalyDetectionAdapter → Statistical models                        │
│  DecisionEngineAdapter → Multi-signal aggregation                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Dependency Direction (Enforced)

```
interfaces → application → domain ← infrastructure
```

The **domain layer never imports** FastAPI, SQLAlchemy, or any external framework.

### Data Architecture (Medallion Pattern)

```
Raw CSV/TXT ──▶ Bronze (immutable) ──▶ Silver (clean) ──▶ Gold (ML-ready)
  data/raw/       data/bronze/           data/silver/       data/gold/
                  Parquet partitioned     Validated          Train/Val/Test splits
                  by code={TICKER}        Type-coerced       50+ features + targets
```

---

## Project Structure

```
fixtrade/
│
├── app/                           # FastAPI application (Hexagonal Architecture)
│   ├── main.py                    # Application entry point & composition root
│   ├── wsgi.py                    # WSGI compatibility layer (Gunicorn/Waitress)
│   ├── core/
│   │   ├── config.py              # Centralized settings (Pydantic, .env)
│   │   └── db.py                  # Database connection management
│   │
│   ├── domain/trading/            # Pure business logic (no framework imports)
│   │   ├── entities.py            # StockPrice, PricePrediction, Portfolio, etc.
│   │   ├── errors.py              # Domain-specific exceptions
│   │   └── ports.py               # Abstract port interfaces (ABCs)
│   │
│   ├── application/trading/       # Use case orchestration
│   │   ├── predict_price.py       # PredictPriceUseCase
│   │   ├── get_sentiment.py       # GetSentimentUseCase
│   │   ├── detect_anomalies.py    # DetectAnomaliesUseCase
│   │   ├── get_recommendation.py  # GetRecommendationUseCase
│   │   └── dtos.py                # Input/Output data transfer objects
│   │
│   ├── infrastructure/trading/    # Concrete adapter implementations
│   │   ├── price_prediction_adapter.py
│   │   ├── sentiment_analysis_adapter.py
│   │   ├── anomaly_detection_adapter.py
│   │   ├── decision_engine_adapter.py
│   │   ├── stock_price_repository.py
│   │   └── portfolio_repository.py
│   │
│   ├── interfaces/trading/        # FastAPI routes & schemas
│   │   ├── router.py              # REST endpoints
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # Dependency injection wiring
│   │
│   ├── interfaces/health.py       # Health check endpoint
│   │
│   ├── nlp/                       # NLP sentiment analysis service
│   │   ├── sentiment.py           # SentimentAnalyzer (XLM-RoBERTa)
│   │   ├── lowconfidenceerror.py   # Low confidence threshold error
│   │   └── unknownlabelserror.py   # Unknown label mapping error
│   │
│   └── shared/                    # Cross-cutting concerns
│       ├── logging.py             # Structured logging configuration
│       ├── errors/handlers.py     # Centralized domain→HTTP error mapping
│       └── security/
│           ├── headers.py         # Security headers middleware
│           └── rate_limiting.py   # slowapi rate limiter
│
├── prediction/                    # ML prediction module
│   ├── __main__.py                # python -m prediction entrypoint
│   ├── cli.py                     # CLI: etl, train, predict, warm-cache, mlflow-ui
│   ├── config.py                  # ML configuration (6 sub-configs)
│   ├── pipeline.py                # ETL pipeline orchestrator
│   ├── training.py                # Walk-forward CV training with MLflow
│   ├── inference.py               # Prediction service (cache → model → result)
│   ├── etl/
│   │   ├── extract/bvmt_extractor.py      # CSV/TXT ingestion
│   │   ├── transform/bronze_to_silver.py  # Validation & cleaning
│   │   ├── transform/silver_to_gold.py    # Targets & train/val/test splits
│   │   └── load/parquet_loader.py         # Partitioned Parquet I/O
│   ├── features/
│   │   ├── pipeline.py            # Feature orchestrator
│   │   ├── technical.py           # SMA, EMA, RSI, MACD, Bollinger, ATR (27 features)
│   │   ├── temporal.py            # Calendar, cyclical encoding, holidays (16 features)
│   │   ├── volume.py              # VWAP, MFI, A/D line, ratios (8 features)
│   │   └── lag.py                 # Price/return lags, momentum, drawdown (15+ features)
│   ├── models/
│   │   ├── base.py                # BasePredictionModel ABC (Strategy Pattern)
│   │   ├── lstm.py                # PyTorch stacked LSTM
│   │   ├── xgboost_model.py       # XGBoost gradient-boosted trees
│   │   ├── prophet_model.py       # Facebook Prophet (trend + seasonality)
│   │   └── ensemble.py            # Weighted ensemble + liquidity tiers
│   └── utils/
│       ├── cache.py               # Redis cache client + in-memory fallback
│       └── metrics.py             # ModelMonitor, drift detection, alerts
│
├── scraping/                      # Scrapy news article collection
│   ├── items.py                   # ArticleItem definition
│   ├── pipelines.py               # PostgresPipeline (DB + JSONL fallback)
│   ├── settings.py                # Scrapy configuration
│   ├── utils.py                   # Multi-locale date parsing
│   └── spiders/
│       ├── millim_spider.py       # Millim.tn financial news spider
│       ├── ilboursa_spider.py     # IlBoursa.com market news spider
│       └── example_spider.py      # Template spider
│
├── data/                          # Medallion data architecture
│   ├── raw/                       # Source CSV/TXT files
│   ├── bronze/                    # Immutable partitioned Parquet
│   ├── silver/                    # Validated & cleaned
│   └── gold/                      # ML-ready train/val/test
│
├── db/
│   ├── 001_init_schema.sql        # Full PostgreSQL schema (10 tables + 3 views)
│   └── load_data.py               # Bulk CSV → PostgreSQL loader
│
├── models/                        # Trained model artifacts
│   └── ensemble/                  # Ensemble weights & per-ticker models
│
├── tests/
│   ├── test_api_trading.py        # API endpoint tests
│   ├── test_application_trading.py # Use case tests
│   ├── test_domain_trading.py     # Domain entity & error tests
│   └── test_prediction.py         # ML pipeline tests (60 tests)
│
├── docs/
│   ├── DATA_FLOW.md               # Row-level data journey walkthrough
│   └── PREDICTION_MODULE.md       # Full ML module documentation
│
├── scripts/
│   └── check_db_connect.py        # Database connectivity check
│
├── docker/
│   └── Dockerfile                 # Multi-stage production image
├── Dockerfile                     # Development image
├── docker-compose.yml             # Full stack: API + PostgreSQL + Redis
├── requirements.txt               # Pinned Python dependencies
├── run_training.py                # One-command full training pipeline
└── scrapy.cfg                     # Scrapy project configuration
```

---

## Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| **Web Framework** | FastAPI 0.115 | REST API with async support |
| **ASGI Server** | Uvicorn 0.30 | Development & production server |
| **WSGI Bridge** | asgiref 3.8 | Legacy server compatibility |
| **Production Server** | Gunicorn 23.0 | Multi-worker WSGI/ASGI deployment |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **Settings** | pydantic-settings 2.5 | Environment-based configuration |
| **Database** | PostgreSQL 16 | Relational data store (market data, portfolios) |
| **ORM** | SQLAlchemy 2.0 | Database toolkit |
| **DB Driver** | pg8000 1.29 | Pure-Python PostgreSQL driver |
| **Cache** | Redis 7 | Prediction cache & feature store |
| **Rate Limiting** | slowapi 0.1.9 | Per-endpoint rate limiting |
| **ML: Deep Learning** | PyTorch 2.1+ | LSTM neural network |
| **ML: Gradient Boosting** | XGBoost 2.0+ | Tree-based ensemble model |
| **ML: Time Series** | Prophet 1.1+ | Trend & seasonality decomposition |
| **ML: Preprocessing** | scikit-learn 1.4+ | MinMaxScaler, metrics |
| **Data Processing** | pandas 2.1+ / NumPy 1.26+ | DataFrame operations, ETL |
| **Storage** | PyArrow 14+ | Parquet columnar storage (~80% compression) |
| **NLP** | HuggingFace Transformers 4.51 | XLM-RoBERTa sentiment analysis |
| **Experiment Tracking** | MLflow 2.10+ | Model registry, metric comparison |
| **Web Scraping** | Scrapy 2.11 | News article collection |
| **Date Parsing** | dateparser 1.1 / python-dateutil 2.8 | Multi-locale date handling |
| **Testing** | pytest 8.3 / httpx 0.27 | Test runner & async HTTP client |
| **Containerization** | Docker + Docker Compose | Reproducible deployment |

---

## Prerequisites

- **Python** 3.11+
- **PostgreSQL** 16+ (or use Docker)
- **Redis** 7+ (or use Docker)
- **Docker** & **Docker Compose** (recommended for infrastructure)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/obeidallahnouicer/fixtrade.git
cd fixtrade
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials, Redis URL, and other settings
```

### 5. Start Infrastructure (Docker)

```bash
docker compose up -d postgres redis
```

### 6. Initialize Database Schema

The schema is automatically applied on PostgreSQL container startup via `db/001_init_schema.sql`. To verify:

```bash
python scripts/check_db_connect.py
```

---

## Configuration

All configuration is managed through environment variables loaded via `.env`:

| Variable | Description | Default |
|---|---|---|
| `PROJECT_NAME` | Application display name | `FixTrade` |
| `VERSION` | Application version | `1.0.0` |
| `DEBUG` | Enable debug mode (exposes /docs, /redoc) | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `POSTGRES_HOST` | PostgreSQL hostname | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | PostgreSQL database name | `fixtrade` |
| `POSTGRES_USER` | PostgreSQL username | `fixtrade` |
| `POSTGRES_PASSWORD` | PostgreSQL password | — |
| `DATABASE_URL` | Full PostgreSQL connection string | — |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database number | `0` |
| `REDIS_PASSWORD` | Redis password | — |
| `REDIS_URL` | Full Redis connection string | — |
| `FIXTRADE_DATA_DIR` | Path to data directory | `data` |
| `MODEL_DIR` | Path to trained model artifacts | `models` |
| `PREDICTION_CACHE_TTL` | Prediction cache TTL (seconds) | `3600` |
| `RATE_LIMIT_DEFAULT` | Default rate limit | `60/minute` |
| `RATE_LIMIT_HEAVY` | Rate limit for compute-heavy endpoints | `10/minute` |
| `MLFLOW_TRACKING_URI` | MLflow tracking backend | `mlruns` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `fixtrade-prediction` |
| `SCRAPING_POSTGRES_DSN` | Scraping pipeline DB connection | — |

---

## Running the Application

### Development (ASGI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production (Gunicorn)

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Production (WSGI Fallback)

```bash
gunicorn app.wsgi:application -w 4 --bind 0.0.0.0:8000
```

### Full Stack (Docker Compose)

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** on port 5432 (with schema auto-initialization)
- **Redis** on port 6379 (256 MB, LRU eviction)
- **FastAPI** on port 8000 (with health checks)

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Health Check

```
GET /api/v1/health
```

**Response** `200 OK`:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Price Prediction

```
POST /api/v1/trading/predictions
```

**Request Body**:
```json
{
  "symbol": "BIAT",
  "horizon_days": 3
}
```

| Field | Type | Constraints |
|---|---|---|
| `symbol` | string | 2–10 chars, uppercase alphanumeric, pattern `^[A-Z0-9]+$` |
| `horizon_days` | integer | 1–5 |

**Response** `200 OK`:
```json
{
  "predictions": [
    {
      "symbol": "BIAT",
      "target_date": "2026-02-09",
      "predicted_close": 97.250,
      "confidence_lower": 95.100,
      "confidence_upper": 99.400
    }
  ]
}
```

### Sentiment Analysis

```
POST /api/v1/trading/sentiment
```

**Request Body**:
```json
{
  "symbol": "SFBT",
  "target_date": "2026-02-08"
}
```

| Field | Type | Constraints |
|---|---|---|
| `symbol` | string | 2–10 chars, uppercase alphanumeric |
| `target_date` | date (ISO 8601) | Optional, defaults to today |

**Response** `200 OK`:
```json
{
  "symbol": "SFBT",
  "date": "2026-02-08",
  "score": 0.72,
  "sentiment": "positive",
  "article_count": 5
}
```

### Anomaly Detection

```
POST /api/v1/trading/anomalies
```

**Request Body**:
```json
{
  "symbol": "DELICE"
}
```

**Response** `200 OK`:
```json
{
  "anomalies": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "symbol": "DELICE",
      "anomaly_type": "volume_spike",
      "severity": 0.87,
      "description": "Volume 3.2x above 20-day average"
    }
  ]
}
```

### Trade Recommendation

```
POST /api/v1/trading/recommendations
```

**Request Body**:
```json
{
  "symbol": "BIAT",
  "portfolio_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response** `200 OK`:
```json
{
  "symbol": "BIAT",
  "action": "buy",
  "confidence": 0.82,
  "reasoning": "Strong upward price prediction with positive sentiment and no anomalies detected"
}
```

### Error Responses

All errors follow a consistent schema:

```json
{
  "error": "Symbol not found",
  "detail": null
}
```

| Status | Error | Condition |
|---|---|---|
| `400` | Insufficient liquidity / funds | Business rule violation |
| `404` | Symbol / Portfolio not found | Resource does not exist |
| `422` | Invalid prediction horizon | Validation failure |
| `429` | Rate limit exceeded | Too many requests |
| `500` | Anomaly detection failed | Internal processing error |

---

## Data Pipeline (ETL)

The ETL pipeline follows the **Medallion Architecture** (Bronze → Silver → Gold):

```
┌─────────────┐     ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌────────────┐
│  RAW CSV    │────▶│ BRONZE  │────▶│  SILVER  │────▶│ FEATURES │────▶│    GOLD    │
│  data/raw/  │     │ (as-is) │     │ (clean)  │     │ (50+ cols)│    │(train/val/ │
│             │     │         │     │          │     │           │    │ test+target)│
└─────────────┘     └─────────┘     └──────────┘     └──────────┘     └────────────┘
```

### Layer Details

| Layer | Format | Purpose | Key File |
|---|---|---|---|
| **Raw** | CSV/TXT | Source BVMT data files | `data/raw/` |
| **Bronze** | Partitioned Parquet | Immutable raw storage, schema-on-read | `bvmt_extractor.py` |
| **Silver** | Partitioned Parquet | Validated, cleaned, type-coerced | `bronze_to_silver.py` |
| **Gold** | DataFrame | ML-ready with targets, chronological splits | `silver_to_gold.py` |

### Data Quality Rules (Silver Layer)

| Rule | Logic | Catches |
|---|---|---|
| `cloture_positive` | `cloture > 0` | Zero/negative closing prices |
| `high_gte_low` | `plus_haut >= plus_bas` | Swapped high/low values |
| `volume_non_negative` | `quantite_negociee >= 0` | Negative volume entries |
| `no_future_dates` | `seance <= today` | Future-dated rows |

### Running the Pipeline

```bash
# Full ETL pipeline
python -m prediction etl

# Incremental (only new data since last watermark)
python -m prediction etl --incremental
```

> For a row-level walkthrough of data transformations, see [docs/DATA_FLOW.md](docs/DATA_FLOW.md).

---

## Feature Engineering

The feature pipeline generates **50+ features** per stock per day, processing each ticker independently to avoid cross-contamination. All lagged features are shifted by 1 day to prevent data leakage.

### Feature Groups

| Group | Features | Module | Count |
|---|---|---|---|
| **Technical Indicators** | SMA (5/10/20/50/200), EMA (12/26), RSI, MACD, Bollinger Bands, ATR, Stochastic, ROC, OBV, Price/SMA ratios, Volatility | `features/technical.py` | 27 |
| **Temporal / Calendar** | Day-of-week, month, quarter, cyclical sin/cos encoding, Tunisian holidays, trading day of month | `features/temporal.py` | 16 |
| **Volume Profile** | Volume SMAs, volume ratio, volume trend, VWAP, Price/VWAP, A/D line, MFI | `features/volume.py` | 8 |
| **Lag & Momentum** | Price lags (1–20d), return lags, rolling return stats (mean/std/skew), cumulative returns, max drawdown, momentum, mean reversion z-score | `features/lag.py` | 15+ |

### Tunisian Holidays Encoded

New Year, Revolution Day (Jan 14), Independence Day (Mar 20), Martyrs' Day (Apr 9), Labour Day (May 1), Victory Day (Jun 1), Republic Day (Jul 25), Women's Day (Aug 13), Evacuation Day (Oct 15).

### Anti-Leakage Guarantee

Every technical, volume, and lag feature uses `.shift(1)` — the value at day *t* is computed **only** from data up to day *t−1*. Gold-layer splits are strictly chronological (no random shuffling).

---

## ML Models & Training

### Model Architecture

Three base models combined via a weighted ensemble:

| Model | Strengths | Default Weight |
|---|---|---|
| **LSTM** (PyTorch) | Temporal dependencies, sequence patterns | 0.45 |
| **XGBoost** | Feature interactions, tabular patterns, missing values | 0.35 |
| **Prophet** | Long-term trends, seasonality, low-liquidity stocks | 0.20 |

### Liquidity-Tiered Model Selection

| Tier | Volume Threshold | Models Used |
|---|---|---|
| Tier 1 (High) | > 10,000 daily avg | LSTM + XGBoost + Prophet |
| Tier 2 (Medium) | 1,000–10,000 | XGBoost + Prophet |
| Tier 3 (Low) | < 1,000 | Prophet only |

### Training Pipeline

Walk-forward cross-validation with 3 chronological splits:

| Split | Train Period | Validation | Test |
|---|---|---|---|
| Split 1 | ≤ 2022 | 2023 | 2024 |
| Split 2 | ≤ 2023 | 2024 | 2025 |
| Split 3 | ≤ 2024 | 2025 | — |

All training runs are tracked in **MLflow** with logged hyperparameters and metrics (MAE, RMSE, MAPE, Directional Accuracy, R²).

### Model Hyperparameters

| LSTM | Value |
|---|---|
| Sequence length | 30 |
| Hidden size | 64 |
| Layers | 2 |
| Dropout | 0.3 |
| Learning rate | 0.002 |
| Epochs | 50 |
| Batch size | 512 |
| Early stopping patience | 8 |

| XGBoost | Value |
|---|---|
| Estimators | 400 |
| Max depth | 7 |
| Learning rate | 0.03 |
| Subsample | 0.85 |
| Column sample | 0.75 |
| Early stopping rounds | 20 |

| Prophet | Value |
|---|---|
| Changepoint prior scale | 0.1 |
| Seasonality prior scale | 5.0 |
| Yearly seasonality | Yes |
| Weekly seasonality | Yes |

### Running Training

```bash
# Full pipeline: ETL → Walk-Forward CV → Final model → MLflow UI
python run_training.py

# Skip ETL (use existing processed data)
python run_training.py --skip-etl

# Train only final production model
python run_training.py --final-only

# Train a specific ticker
python -m prediction train --symbol BIAT

# Train top N most liquid tickers
python -m prediction train --top-n 10
```

---

## Inference & Caching

### Inference Flow

```
Request ──▶ Check Redis Cache ──▶ [HIT] Return cached ──▶ Response
                   │
                [MISS]
                   │
                   ▼
            Load Ensemble Model
            (per-symbol → global fallback)
                   │
                   ▼
            Fetch Latest Features
            (Silver layer Parquet)
                   │
                   ▼
            Run Ensemble Inference
            (LSTM + XGBoost + Prophet)
                   │
                   ▼
            Compute Confidence Intervals
                   │
                   ▼
            Cache Result (TTL-based)
                   │
                   ▼
                Response
```

### Caching Strategy

| Key Pattern | TTL | Description |
|---|---|---|
| `pred:{ticker}:{model}` | 1 hour (intraday) | Prediction results |
| `pred:{ticker}:{model}` | 12 hours (post-market) | After market close |
| `feat:{ticker}:{date}` | 90 days | Feature store |

Redis uses **LRU eviction** with a 256 MB memory limit. If Redis is unavailable, an in-memory dict provides a graceful fallback.

### Cache Warming

```bash
# Pre-compute predictions for top 30 tickers
python -m prediction warm-cache
```

---

## NLP Sentiment Analysis

### Model

**XLM-RoBERTa** (`jplu/tf-xlm-roberta-large`) — a multilingual transformer supporting 100+ languages, including French and Arabic, ideal for analyzing BVMT-related news.

### Label Mapping

| Model Output | Mapped Score |
|---|---|
| `positive` | `+1` |
| `negative` | `-1` |
| `neutral` | `0` |

### Text Preprocessing

1. Unicode NFC normalization
2. Arabic diacritics (tashkeel) removal
3. Emoji and symbol removal
4. Whitespace normalization

### Usage

```python
from app.nlp.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Single article
score = analyzer.analyze("La société XYZ a annoncé une augmentation de 20% de son chiffre d'affaires.")
# Returns: 1 (positive)

# Batch analysis (parallelized)
scores = analyzer.analyze_batch([
    "Les bénéfices ont fortement augmenté.",
    "أعلنت الشركة عن انخفاض حاد في أرباحها الفصلية.",
    "Le marché reste stable aujourd'hui.",
])
# Returns: [1, -1, 0]
```

### Error Handling

| Error | Condition |
|---|---|
| `UnknownLabelError` | Model returns a label not in the expected mapping |
| `LowConfidenceError` | Model confidence falls below configurable threshold |
| `ValueError` | Empty or blank input text |

---

## Web Scraping

### Spiders

| Spider | Source | Target Content |
|---|---|---|
| `millim` | [millim.tn](https://www.millim.tn) | Financial news articles |
| `ilboursa` | [ilboursa.com](https://www.ilboursa.com) | Market news, magazine articles |

### Pipeline

Scraped articles flow through the `PostgresPipeline`:

1. Parse article fields (title, URL, date, summary, content)
2. Normalize dates using multi-locale parsing (French, Arabic, English)
3. Insert into `scraped_articles` table with deduplication (unique on URL)
4. **Fallback**: If PostgreSQL is unreachable, articles are written to a local JSONL file

### Running Spiders

```bash
# Run a spider
scrapy crawl millim

# Run via Docker Compose
docker compose run scraper scrapy crawl millim
```

### Configuration

- **Polite crawling**: `DOWNLOAD_DELAY=1.0`, `CONCURRENT_REQUESTS=8`
- **robots.txt**: Obeyed (`ROBOTSTXT_OBEY=True`)
- **User agent**: `fixtrade-scraper/1.0`

---

## Database Schema

PostgreSQL schema with **10 tables** and **3 views** (see `db/001_init_schema.sql`):

### Tables

| Table | Purpose |
|---|---|
| `stock_prices` | Historical OHLCV data (indexed by symbol + date) |
| `price_predictions` | Model predictions with confidence intervals |
| `sentiment_scores` | Daily aggregated sentiment per symbol |
| `anomaly_alerts` | Detected market anomalies |
| `portfolios` | Virtual portfolio definitions |
| `portfolio_positions` | Individual positions within portfolios |
| `trade_recommendations` | Generated buy/sell/hold signals |
| `model_registry` | Trained model metadata and metrics |
| `etl_watermarks` | Incremental ETL progress tracking |
| `scraped_articles` | Raw scraped news articles |

### Views

| View | Description |
|---|---|
| `v_latest_prices` | Most recent price per stock |
| `v_latest_predictions` | Most recent prediction per stock |
| `v_portfolio_value` | Live portfolio valuation with P&L % |

### Bulk Data Loading

```bash
# Load all CSV/TXT files from data/raw/ into PostgreSQL
python db/load_data.py
```

---

## Testing

### Test Structure

| File | Layer | Scope |
|---|---|---|
| `tests/test_domain_trading.py` | Domain | Entities, errors (no IO) |
| `tests/test_application_trading.py` | Application | Use cases with mocked ports |
| `tests/test_api_trading.py` | Interfaces | API routes, validation, security |
| `tests/test_prediction.py` | Prediction | Config, features, ETL, models, cache, anti-leakage (60 tests) |

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_prediction.py

# Verbose with output
pytest -v -s

# With coverage
pytest --cov=app --cov=prediction
```

---

## Docker Deployment

### Production Dockerfile (`docker/Dockerfile`)

Multi-stage build with:
- **Stage 1**: Install dependencies in a builder image
- **Stage 2**: Copy installed packages into a slim production image
- Non-root user (`appuser`) for security
- Pre-created data directories

### Docker Compose Services

```yaml
services:
  postgres:   # PostgreSQL 16 Alpine, auto-schema init, health checks
  redis:      # Redis 7 Alpine, 256 MB LRU, health checks
  api:        # FastAPI application, depends on postgres + redis
  scraper:    # Scrapy spider runner, depends on postgres
```

### Commands

```bash
# Start full stack
docker compose up -d

# View logs
docker compose logs -f api

# Run ETL inside container
docker compose exec api python -m prediction etl

# Run training inside container
docker compose exec api python run_training.py

# Stop all services
docker compose down
```

---

## CLI Reference

### Prediction Module CLI

```bash
python -m prediction <command> [options]
```

| Command | Description | Options |
|---|---|---|
| `etl` | Run the ETL pipeline | `--incremental` |
| `train` | Train ML models | `--symbol BIAT`, `--final`, `--top-n 10` |
| `predict` | Run a single prediction | `--symbol BIAT`, `--days 3`, `--model ensemble` |
| `warm-cache` | Pre-compute predictions for top tickers | — |
| `mlflow-ui` | Launch MLflow tracking dashboard | `--port 5000` |

### Training Runner

```bash
python run_training.py [options]
```

| Option | Description |
|---|---|
| `--skip-etl` | Use existing Silver-layer data |
| `--final-only` | Skip CV, train final production model only |
| `--no-ui` | Don't launch MLflow UI after training |
| `--port 5000` | MLflow UI port |

---

## Security

### Implemented Measures

| Category | Implementation |
|---|---|
| **Input Validation** | Pydantic models with strict typing, min/max constraints, regex patterns |
| **Rate Limiting** | slowapi: 60 req/min default, 10 req/min for heavy endpoints |
| **Security Headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Content-Security-Policy: default-src 'self'`, `X-XSS-Protection: 1; mode=block` |
| **Error Handling** | No stack traces or internal details exposed to clients |
| **SQL Safety** | SQLAlchemy ORM only, parameterized queries, no raw SQL concatenation |
| **Rate Exceeded Response** | HTTP 429 with JSON error body |
| **Container Security** | Non-root user, read-only filesystem, minimal base image |
| **Dependency Pinning** | All versions pinned in `requirements.txt` |

### Explicitly Out of Scope

Authentication, authorization, and user management are **intentionally not implemented** in this MVP. All endpoints are publicly accessible.

---

## Monitoring & Observability

### Structured Logging

All log entries follow a consistent format:

```
2026-02-08 14:30:00 | INFO     | app.main | Application started
```

Logs are written to stdout for container-native log aggregation. Sensitive data (request bodies, secrets) is never logged.

### MLflow Experiment Tracking

All training runs log to MLflow:
- **Parameters**: Model hyperparameters, feature counts, CV splits
- **Metrics**: MAE, RMSE, MAPE, Directional Accuracy, R² per split
- **Artifacts**: Saved model weights, ensemble configuration

Launch the dashboard:

```bash
python -m prediction mlflow-ui --port 5000
# Open http://127.0.0.1:5000
```

### Model Monitoring

The `ModelMonitor` class tracks prediction performance over time and triggers alerts when:
- RMSE exceeds 1.5× the baseline threshold
- Directional accuracy drops below 50%
- Prediction drift is detected over a 30-day rolling window

### Health Check

```
GET /api/v1/health → { "status": "ok", "version": "1.0.0" }
```

Docker Compose includes health checks for all services (PostgreSQL, Redis, API) with configurable intervals and retry policies.

---

## Contributing

1. Follow the **Hexagonal Architecture** layering strictly
2. Domain layer must remain pure — no framework imports
3. One use case per class, one responsibility per function
4. Max 200 lines per file, early returns, no deep nesting
5. All inputs validated with Pydantic, all errors are domain-specific
6. Tests must pass without database or network access (mock ports)
7. Every public class/function must have a docstring

See [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for the full development guidelines.

---

## License

This project is proprietary. All rights reserved.

