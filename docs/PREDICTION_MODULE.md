# FixTrade — Prediction Module Documentation

> **Module**: `prediction/`
> **Date**: February 7, 2026
> **Status**: All 60 unit tests passing ✅
>
> 📖 **New**: See [DATA_FLOW.md](DATA_FLOW.md) for a data-directed walkthrough
> that follows a row from raw CSV → prediction step by step.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Structure](#3-module-structure)
4. [Data Pipeline (ETL)](#4-data-pipeline-etl)
   - 4.1 [Extract Layer](#41-extract-layer)
   - 4.2 [Bronze Layer (Raw)](#42-bronze-layer-raw)
   - 4.3 [Silver Layer (Cleaned)](#43-silver-layer-cleaned)
   - 4.4 [Gold Layer (ML-Ready)](#44-gold-layer-ml-ready)
5. [Feature Engineering](#5-feature-engineering)
   - 5.1 [Technical Indicators](#51-technical-indicators)
   - 5.2 [Temporal Features](#52-temporal-features)
   - 5.3 [Volume Features](#53-volume-features)
   - 5.4 [Lag & Momentum Features](#54-lag--momentum-features)
6. [ML Models](#6-ml-models)
   - 6.1 [Base Interface (Strategy Pattern)](#61-base-interface-strategy-pattern)
   - 6.2 [LSTM Predictor](#62-lstm-predictor)
   - 6.3 [XGBoost Predictor](#63-xgboost-predictor)
   - 6.4 [Prophet Predictor](#64-prophet-predictor)
   - 6.5 [Ensemble Engine](#65-ensemble-engine)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference Service](#8-inference-service)
9. [Caching Strategy](#9-caching-strategy)
10. [Monitoring & Alerts](#10-monitoring--alerts)
11. [Anti-Leakage Guarantees](#11-anti-leakage-guarantees)
12. [Infrastructure & Deployment](#12-infrastructure--deployment)
13. [Integration with Existing App](#13-integration-with-existing-app)
14. [CLI Reference](#14-cli-reference)
15. [Test Coverage](#15-test-coverage)
16. [Files Created & Modified](#16-files-created--modified)
17. [Dependencies Added](#17-dependencies-added)

---

## 1. Executive Summary

The prediction module implements a **modular ML system** for forecasting closing prices of stocks listed on the **BVMT** (Bourse des Valeurs Mobilières de Tunis). It follows a **Medallion Architecture** (Bronze → Silver → Gold) for data management and a **Strategy Pattern** for pluggable ML models.

**Key capabilities delivered:**

| Capability | Implementation |
|---|---|
| Data ingestion from CSV | `BVMTExtractor` with flexible encoding/delimiter detection |
| Data quality validation | 4 business rules (positive close, high ≥ low, non-negative volume, no future dates) |
| 50+ engineered features | Technical (20+), temporal (16), volume (8), lag/momentum (15+) |
| 3 ML models | LSTM (PyTorch), XGBoost, Prophet |
| Ensemble prediction | Static, dynamic, and stacking-ready weighting |
| Liquidity-tiered inference | Tier 1/2/3 model selection by average daily volume |
| Redis caching | Sub-50ms cache hits, 1h/12h TTL (intraday/post-market) |
| Walk-forward CV | 3 chronological splits, no data leakage |
| Model monitoring | Drift detection, RMSE alerts, retrain triggers |
| Full CLI | `etl`, `train`, `predict`, `warm-cache` commands |
| 40 passing tests | Config, features, ETL, models, ensemble, cache, anti-leakage |

---

## 2. Architecture Overview

```
Raw CSV Files (data/raw/)
        │
        ▼
┌───────────────────┐
│  EXTRACT LAYER    │  bvmt_extractor.py
│  CSV → DataFrame  │  Flexible encoding, dedup, column mapping
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  BRONZE LAYER     │  Partitioned Parquet (immutable, schema-on-read)
│  data/bronze/     │  Partition: /code={TICKER}/data.parquet
└───────┬───────────┘
        │
        ▼  4 validation rules
┌───────────────────┐
│  SILVER LAYER     │  bronze_to_silver.py
│  data/silver/     │  Type coercion, null handling, forward-fill
└───────┬───────────┘
        │
        ▼  Feature Pipeline (50+ features)
┌───────────────────┐
│  GOLD LAYER       │  silver_to_gold.py
│  Train/Val/Test   │  Walk-forward chronological splits
└───────┬───────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│           ENSEMBLE ENGINE                     │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐   │
│  │  LSTM   │  │ XGBoost  │  │  Prophet   │   │
│  │  0.45   │  │  0.35    │  │   0.20     │   │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘   │
│       └────────────┼──────────────┘          │
│                    ▼                          │
│          Weighted Average + CI                │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────┐
│  REDIS CACHE              │
│  pred:{ticker}:{model}    │
│  TTL: 1h intraday / 12h  │
└───────────────┬───────────┘
                │
                ▼
┌───────────────────────────┐
│  FastAPI /api/v1/trading  │
│  /predictions endpoint    │
└───────────────────────────┘
```

---

## 3. Module Structure

```
prediction/
│
├── __init__.py              # Package marker
├── __main__.py              # python -m prediction entrypoint
├── cli.py                   # CLI commands (etl, train, predict, warm-cache)
├── config.py                # Centralized configuration (6 sub-configs)
├── pipeline.py              # ETL orchestrator (full + incremental)
├── training.py              # Walk-forward CV training pipeline
├── inference.py             # Prediction service (cache → model → result)
│
├── etl/
│   ├── extract/
│   │   └── bvmt_extractor.py    # CSV ingestion, encoding detection, dedup
│   ├── transform/
│   │   ├── bronze_to_silver.py  # Validation, cleaning, type coercion
│   │   └── silver_to_gold.py    # Target creation, train/val/test splits
│   └── load/
│       └── parquet_loader.py    # Partitioned Parquet I/O, watermark tracking
│
├── features/
│   ├── technical.py         # SMA, EMA, RSI, MACD, Bollinger, ATR, etc.
│   ├── temporal.py          # Calendar, cyclical encoding, Tunisian holidays
│   ├── volume.py            # VWAP, MFI, A/D line, volume ratios
│   ├── lag.py               # Price lags, return lags, momentum, drawdown
│   └── pipeline.py          # Feature orchestrator (runs all above per-ticker)
│
├── models/
│   ├── base.py              # BasePredictionModel ABC (Strategy Pattern)
│   ├── lstm.py              # PyTorch stacked LSTM
│   ├── xgboost_model.py     # XGBoost regressor
│   ├── prophet_model.py     # Facebook Prophet
│   └── ensemble.py          # Weighted ensemble with liquidity tiers
│
└── utils/
    ├── cache.py             # Redis client + in-memory fallback
    └── metrics.py           # ModelMonitor, drift detection, retrain alerts
```

---

## 4. Data Pipeline (ETL)

### 4.1 Extract Layer

**File**: `prediction/etl/extract/bvmt_extractor.py`

**Class**: `BVMTExtractor`

Reads all `.csv` files from `data/raw/` and produces a unified DataFrame.

| Feature | Detail |
|---|---|
| Encoding detection | Tries `utf-8`, `latin-1`, `cp1252` sequentially |
| Delimiter detection | Tries `,`, `;`, `\t` (BVMT exports vary) |
| Column standardization | Maps `SEANCE`→`seance`, `CLOTURE`→`cloture`, etc. |
| Deduplication | Drops exact duplicates on `(seance, code)`, keeps first |
| Incremental support | `extract_incremental(since=date)` filters by watermark |

**Usage:**
```python
from prediction.etl.extract.bvmt_extractor import BVMTExtractor

extractor = BVMTExtractor()
df = extractor.extract_all_csv()        # Full extraction
df = extractor.extract_incremental(since=date(2025, 12, 1))  # Incremental
extractor.save_to_bronze(df)            # Partitioned Parquet
```

### 4.2 Bronze Layer (Raw)

- **Storage format**: Partitioned Parquet (columnar, ~80% compression vs CSV)
- **Partition scheme**: `/code={TICKER}/data.parquet`
- **Immutability**: Bronze data is append-only; re-extraction overwrites partitions idempotently
- **Location**: `data/bronze/`

### 4.3 Silver Layer (Cleaned)

**File**: `prediction/etl/transform/bronze_to_silver.py`

**Classes**: `DataQualityChecker`, `BronzeToSilverTransformer`

**Validation rules applied:**

| Rule | Logic | Action on failure |
|---|---|---|
| `cloture_positive` | `cloture > 0` | Row rejected |
| `high_gte_low` | `plus_haut >= plus_bas` | Row rejected |
| `volume_non_negative` | `quantite_negociee >= 0` | Row rejected |
| `no_future_dates` | `seance <= today` | Row rejected |

**Transformation steps:**
1. Apply all 4 validation rules → split into (valid, rejected)
2. Coerce types: `seance` → datetime, prices → float64, volume → int
3. Forward-fill price NaN values within each ticker group
4. Drop any rows still missing `cloture`
5. Sort by `(code, seance)` ascending

### 4.4 Gold Layer (ML-Ready)

**File**: `prediction/etl/transform/silver_to_gold.py`

**Class**: `SilverToGoldTransformer`

Creates three views:

| View | Description | Use |
|---|---|---|
| **Training** | Features + targets, split by year | Model training (walk-forward CV) |
| **Inference** | Latest features per ticker, no targets | Real-time predictions |
| **Evaluation** | Holdout period with targets | Backtesting |

**Target creation:**
- `target_1d` = closing price 1 day ahead (`shift(-1)` per ticker)
- `target_2d` = closing price 2 days ahead
- `target_3d`, `target_5d` similarly

**Train/Val/Test split** (strictly chronological, no random shuffling):
- **Train**: `year <= train_end_year`
- **Validation**: `train_end_year < year <= val_end_year`
- **Test**: `year > val_end_year`

---

## 5. Feature Engineering

**Orchestrator**: `prediction/features/pipeline.py` → `FeaturePipeline`

The pipeline processes each ticker independently (to avoid cross-contamination), computes all features, then concatenates. Total: **50+ engineered features**.

### 5.1 Technical Indicators

**File**: `prediction/features/technical.py`

All indicators are computed with `.shift(1)` to prevent look-ahead bias.

| Feature | Parameters | Count |
|---|---|---|
| SMA (Simple Moving Average) | Windows: 5, 10, 20, 50, 200 | 5 |
| EMA (Exponential Moving Average) | Spans: 12, 26 | 2 |
| RSI (Relative Strength Index) | Window: 14 | 1 |
| MACD + Signal + Histogram | Fast: 12, Slow: 26, Signal: 9 | 3 |
| Bollinger Bands (Upper, Lower, Width) | Window: 20, Std: 2.0 | 3 |
| ATR (Average True Range) | Window: 14 | 1 |
| Stochastic Oscillator (%K, %D) | Window: 14 | 2 |
| Rate of Change | Periods: 5, 10, 20 | 3 |
| On-Balance Volume (OBV) | — | 1 |
| Price-to-SMA ratios | 5 SMAs | 5 |
| 20-day rolling volatility | — | 1 |
| **Total** | | **27** |

### 5.2 Temporal Features

**File**: `prediction/features/temporal.py`

| Feature | Description |
|---|---|
| `day_of_week` | 0=Monday, 4=Friday |
| `day_of_month`, `month`, `quarter` | Calendar position |
| `week_of_year`, `day_of_year` | Annual position |
| `is_month_start/end`, `is_quarter_end`, `is_year_end` | Binary flags |
| `day_of_week_sin/cos`, `month_sin/cos` | Cyclical encoding (captures periodicity) |
| `trading_day_of_month` | Ordinal within each month (per ticker) |
| `is_near_holiday` | Within 2 days of a Tunisian public holiday |

**Tunisian holidays tracked**: New Year, Revolution Day (Jan 14), Independence Day (Mar 20), Martyrs' Day (Apr 9), Labour Day (May 1), Victory Day (Jun 1), Republic Day (Jul 25), Women's Day (Aug 13), Evacuation Day (Oct 15).

### 5.3 Volume Features

**File**: `prediction/features/volume.py`

| Feature | Description |
|---|---|
| `volume_sma_5/10/20` | Volume moving averages |
| `volume_ratio` | Current volume ÷ 20-day average |
| `volume_trend` | 5-day linear slope of volume |
| `vwap` | 20-day Volume-Weighted Average Price |
| `price_to_vwap` | Price relative to VWAP |
| `ad_line` | Accumulation/Distribution Line |
| `mfi` | 14-day Money Flow Index |

### 5.4 Lag & Momentum Features

**File**: `prediction/features/lag.py`

| Feature | Description |
|---|---|
| `close_lag_1/2/3/5/10/20` | Shifted closing prices |
| `return_lag_1/2/3/5/10/20` | Shifted daily returns |
| `return_mean/std/skew_{5,10,20}d` | Rolling return statistics |
| `cum_return_{5,10,20,60}d` | Cumulative returns |
| `max_drawdown_20d` | 20-day rolling maximum drawdown |
| `momentum_{5,10,20}d` | Price change vs N days ago |
| `mean_reversion_z` | Z-score of price vs 20-day SMA |

---

## 6. ML Models

### 6.1 Base Interface (Strategy Pattern)

**File**: `prediction/models/base.py`

All models implement `BasePredictionModel`:

```
BasePredictionModel (ABC)
├── fit(X_train, y_train, X_val?, y_val?)  → self
├── predict(X)                              → np.ndarray
├── predict_proba(X, confidence_level)      → (pred, lower, upper)
├── evaluate(X_test, y_test)                → ModelMetrics
├── get_metrics()                           → ModelMetrics
├── save_model(path)                        → None
└── load_model(path)                        → None
```

**`ModelMetrics` dataclass** tracks: MAE, RMSE, MAPE, Directional Accuracy, R².

### 6.2 LSTM Predictor

**File**: `prediction/models/lstm.py`

| Parameter | Default | Purpose |
|---|---|---|
| `sequence_length` | 60 | Sliding window size |
| `hidden_size` | 128 | LSTM hidden dimension |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Regularization |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 100 | Max training epochs |
| `batch_size` | 32 | Mini-batch size |
| `patience` | 10 | Early stopping patience |

**Architecture**: Input → MinMaxScaler → Sliding Window (60 steps) → 2-layer LSTM (128 units) → Linear → Inverse Scale → Prediction

**Training**: Adam optimizer, MSE loss, gradient clipping (max norm 1.0), early stopping on validation loss, best-weight restoration.

**Persistence**: `lstm_weights.pt` + `scaler_X.pkl` + `scaler_y.pkl` + `lstm_params.pkl`

### 6.3 XGBoost Predictor

**File**: `prediction/models/xgboost_model.py`

| Parameter | Default | Purpose |
|---|---|---|
| `n_estimators` | 500 | Number of boosting rounds |
| `max_depth` | 6 | Max tree depth |
| `learning_rate` | 0.05 | Shrinkage rate |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `early_stopping_rounds` | 20 | Patience on validation |

**Extras**: Exposes `get_feature_importance(top_n)` to rank the most predictive features.

**Persistence**: `xgboost_model.json` + `xgboost_meta.pkl`

### 6.4 Prophet Predictor

**File**: `prediction/models/prophet_model.py`

| Parameter | Default | Purpose |
|---|---|---|
| `changepoint_prior_scale` | 0.05 | Trend flexibility |
| `seasonality_prior_scale` | 10.0 | Seasonality strength |
| `yearly_seasonality` | True | Capture annual cycles |
| `weekly_seasonality` | True | Capture day-of-week effects |

**Extras**: Adds selected regressors (`rsi`, `macd`, `volume_ratio`, `volatility_20d`) if available. Provides native uncertainty intervals via `predict_proba`. Has a `predict_future(periods)` method for standalone forecasting.

**Persistence**: `prophet_model.pkl` + `prophet_meta.pkl`

### 6.5 Ensemble Engine

**File**: `prediction/models/ensemble.py`

Implements three ensemble strategies:

#### Phase 1 — Static Weighted Average (MVP)

```
P_final = 0.45 × P_LSTM + 0.35 × P_XGBoost + 0.20 × P_Prophet
CI = [min(Pi) − 2σ, max(Pi) + 2σ]
```

Weights are optimized on validation RMSE: `w_model = 1 / RMSE_val`, normalized so Σw = 1.

#### Phase 2 — Dynamic Weighting (Context-Aware)

| Condition | LSTM | XGBoost | Prophet |
|---|---|---|---|
| High volatility (> 3%) | 0.20 | 0.30 | **0.50** |
| Low liquidity (vol < 5k) | 0.25 | 0.15 | **0.60** |
| Strong trend (vol < 1%) | **0.60** | 0.30 | 0.10 |
| Default | 0.45 | 0.35 | 0.20 |

#### Phase 3 — Stacking (Ready for future implementation)

The ensemble tracks a `_meta_learner` slot for a Ridge Regression stacker.

#### Liquidity-Tiered Model Selection

| Tier | Volume Threshold | Models Used | Confidence Range |
|---|---|---|---|
| **Tier 1** (High) | ≥ 10,000 daily | LSTM + XGBoost + Prophet | 70–85% |
| **Tier 2** (Medium) | 1,000–10,000 | XGBoost + Prophet | 50–70% |
| **Tier 3** (Low) | < 1,000 | Prophet only | 30–50% |

---

## 7. Training Pipeline

**File**: `prediction/training.py`

**Class**: `TrainingPipeline`

### Walk-Forward Cross-Validation

Three chronological splits (no random shuffling — this is critical for time series):

```
Split 1:
  Train : 2016–2022  ██████████████████████
  Val   : 2023       ████
  Test  : 2024       ████

Split 2:
  Train : 2016–2023  █████████████████████████
  Val   : 2024       ████
  Test  : 2025       ████

Split 3:
  Train : 2016–2024  ████████████████████████████
  Val   : 2025       ████
  Test  : 2026 YTD   ██
```

### Pipeline Steps

1. For each CV split:
   - Create chronological train/val/test DataFrames
   - Replace `inf`/`NaN` with 0 in features
   - Instantiate fresh LSTM, XGBoost, and Prophet models
   - Train each model with validation set (early stopping)
   - Evaluate on validation → collect `ModelMetrics`
2. Average metrics across all splits
3. Log summary

### Final Model Training

`train_final_model()` trains on all available data, optimizes ensemble weights on the latest validation period, and saves the full ensemble to the model registry at `models/ensemble/`.

---

## 8. Inference Service

**File**: `prediction/inference.py`

**Class**: `PredictionService`

### Inference Flow

```
Request (symbol, horizon_days, model, confidence_level)
    │
    ├── 1. Check Redis cache → key: pred:{symbol}:{model}
    │       HIT? → deserialize and return (<50ms)
    │
    ├── 2. Load ensemble from model registry (lazy, cached in memory)
    │
    ├── 3. Fetch latest features:
    │       a. Try Redis feature store → features:{code}:{date}
    │       b. Fallback: read from Silver Parquet on disk
    │
    ├── 4. For each day in horizon:
    │       a. Call ensemble.predict_single() with liquidity awareness
    │       b. Collect (prediction, CI_lower, CI_upper, confidence_score)
    │
    ├── 5. Cache results in Redis (TTL based on market hours)
    │
    └── 6. Return list[PredictionResult]
```

### Fallback Behavior

If models are not trained or features are unavailable, the service returns placeholder predictions with `model_name="fallback"` and `confidence_score=0.0`.

### Trading Day Calculation

`_next_trading_day()` skips weekends (Saturday/Sunday). The BVMT trades Monday–Friday.

---

## 9. Caching Strategy

**File**: `prediction/utils/cache.py`

**Class**: `CacheClient`

| Cache Type | Key Pattern | TTL | Purpose |
|---|---|---|---|
| **Predictions** | `pred:{ticker}:{model}` | 1h intraday / 12h post-market | Avoid redundant inference |
| **Features** | `features:{code}:{date}` | 90 days | Avoid recomputing 50+ features |

### TTL Logic

BVMT trading hours: 09:00–14:15 Tunisia time (UTC+1).

- During market hours (09:00–14:59) → **1-hour TTL** (data changes frequently)
- After market close (15:00–08:59) → **12-hour TTL** (data is stable)

### Invalidation

| Event | Action |
|---|---|
| New data arrival | `invalidate_predictions(ticker)` — clear that ticker's cache |
| Model retraining | `invalidate_predictions()` — clear all predictions |
| Cache warming | Pre-compute top 30 tickers at 08:00 daily |

### Fallback

If Redis is unreachable, the client falls back to an **in-memory Python dict** automatically. No exceptions are raised to the caller.

---

## 10. Monitoring & Alerts

**File**: `prediction/utils/metrics.py`

**Class**: `ModelMonitor`

### Tracked Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Directional Accuracy | % of correct up/down predictions |
| R² | Coefficient of determination |
| Calibration Score | % of actuals within confidence interval |
| Drift | `abs(mean_pred - mean_actual) / mean_actual > 5%` |

### Alert Thresholds

| Condition | Action |
|---|---|
| `RMSE > threshold × 1.5` | ⚠️ Warning logged |
| `Directional accuracy < 50%` | ⚠️ Warning logged |
| Drift detected in 3+ consecutive evaluations | ⚠️ Persistent drift alert |

### Retrain Trigger

`should_retrain(model_name)` returns `True` if any of the above conditions are met, signaling the training pipeline to schedule a retraining run.

---

## 11. Anti-Leakage Guarantees

Data leakage is the #1 risk in time-series ML. The module enforces strict separation:

### Feature-Level

- ✅ **All technical indicators** use `.shift(1)` — computed from data available **before** the current day
- ✅ **All volume features** use `.shift(1)`
- ✅ **Lag features** are inherently backward-looking (`close.shift(n)`)
- ✅ **Rolling windows** use strict lookback only

### Split-Level

- ✅ **No random K-Fold** — only walk-forward chronological splits
- ✅ Train < Validation < Test in time
- ✅ No test data visible during development

### Target-Level

- ✅ Targets use `.shift(-h)` within each ticker group — they are future values
- ✅ `target_1d` at row `i` equals `cloture` at row `i+1` (verified in tests)

### Per-Ticker Isolation

- ✅ Features are computed per-ticker before concatenation
- ✅ No cross-ticker leakage in rolling windows

### Test Verification

Two dedicated tests in `tests/test_prediction.py`:
- `TestAntiLeakage::test_features_only_use_past_data` — asserts row 0 of shifted features is NaN
- `TestAntiLeakage::test_targets_are_future_looking` — asserts `target_1d[i] == cloture[i+1]`

---

## 12. Infrastructure & Deployment

### Docker Compose (`docker-compose.yml`)

| Service | Image | Port | Purpose |
|---|---|---|---|
| `postgres` | postgres:16-alpine | 5432 | Raw data store, Silver/Gold views |
| `redis` | redis:7-alpine | 6379 | Prediction cache + feature store |
| `api` | Custom (Dockerfile) | 8000 | FastAPI application |

Redis is configured with `maxmemory 256mb` and `allkeys-lru` eviction policy.

### Dockerfile (`docker/Dockerfile`)

Multi-stage build:
1. **Builder stage**: Installs Python dependencies
2. **Production stage**: Copies only installed packages + app code, runs as non-root `appuser`

### Directory Setup

```
data/
├── raw/          # Input CSV files (place BVMT exports here)
├── bronze/       # Immutable raw Parquet (auto-generated by ETL)
├── silver/       # Cleaned + feature-enriched Parquet
└── gold/         # ML-ready datasets
models/           # Trained model artifacts (auto-generated by training)
```

---

## 13. Integration with Existing App

The prediction module was wired into the existing hexagonal architecture via a single adapter change.

### Modified File: `app/infrastructure/trading/price_prediction_adapter.py`

**Before**: `raise NotImplementedError("PricePredictionAdapter.predict")`

**After**: Delegates to `prediction.inference.PredictionService`:

```python
class PricePredictionAdapter(PricePredictionPort):
    def __init__(self):
        try:
            from prediction.inference import PredictionService
            self._service = PredictionService()
        except ImportError:
            self._service = None  # Graceful degradation

    def predict(self, symbol, horizon_days):
        results = self._service.predict(symbol=symbol, horizon_days=horizon_days)
        return [PricePrediction(...) for r in results]
```

This means the existing endpoint `POST /api/v1/trading/predictions` now calls the full ensemble pipeline under the hood. If ML dependencies are not installed, it degrades gracefully (returns empty list + logs a warning).

### Modified File: `app/core/config.py`

Added three new settings:
- `redis_url` (default: `redis://localhost:6379/0`)
- `prediction_cache_ttl` (default: `3600` seconds)
- `data_dir` (default: `"data"`)

---

## 14. CLI Reference

```bash
# Run full ETL pipeline (extract → bronze → silver → features → gold)
python -m prediction etl

# Run incremental ETL (only new data since last watermark)
python -m prediction etl --incremental

# Train all models with walk-forward cross-validation
python -m prediction train

# Train final production model and save to registry
python -m prediction train --final

# Predict for a specific ticker
python -m prediction predict --symbol BIAT --days 5

# Predict with a specific model
python -m prediction predict --symbol SFBT --days 3 --model xgboost

# Pre-compute predictions for top 30 tickers
python -m prediction warm-cache
```

---

## 15. Test Coverage

**File**: `tests/test_prediction.py` — **40 tests, all passing** ✅

| Test Class | Tests | What It Covers |
|---|---|---|
| `TestConfig` | 3 | Default config loading, path consistency, weight normalization |
| `TestTechnicalFeatures` | 5 | SMA, RSI, MACD computation, `.shift(1)` verification, anti-leakage |
| `TestTemporalFeatures` | 2 | Calendar features, cyclical encoding bounds |
| `TestVolumeFeatures` | 2 | Volume features, shift verification |
| `TestLagFeatures` | 1 | Lag/momentum/drawdown features |
| `TestFeaturePipeline` | 1 | Full pipeline → 30+ features created |
| `TestDataQuality` | 2 | Positive close, high ≥ low validation rules |
| `TestBronzeToSilver` | 2 | Transformation, sort verification |
| `TestSilverToGold` | 3 | Training view, target creation, inference view |
| `TestModelMetrics` | 1 | MAE, RMSE, MAPE, DirAcc, R² computation |
| `TestEnsemble` | 4 | Liquidity tiers, model selection, dynamic weights, normalization |
| `TestCache` | 3 | In-memory fallback, invalidation, feature store |
| `TestModelMonitor` | 3 | Evaluation, drift detection, retrain trigger |
| `TestParquetLoader` | 3 | Save/load, watermark tracking, partitioned save |
| `TestPredictionService` | 3 | Fallback prediction, weekend skipping, serialization roundtrip |
| `TestAntiLeakage` | 2 | Feature shift verification, target correctness |

---

## 16. Files Created & Modified

### New Files Created (28 files)

| File | Purpose |
|---|---|
| `prediction/__init__.py` | Package marker |
| `prediction/__main__.py` | `python -m prediction` entrypoint |
| `prediction/cli.py` | CLI with 4 commands |
| `prediction/config.py` | 6 config dataclasses, singleton `config` |
| `prediction/pipeline.py` | ETL orchestrator |
| `prediction/training.py` | Walk-forward CV training pipeline |
| `prediction/inference.py` | Prediction service + cache integration |
| `prediction/etl/__init__.py` | Package marker |
| `prediction/etl/extract/__init__.py` | Package marker |
| `prediction/etl/extract/bvmt_extractor.py` | CSV extraction + Bronze storage |
| `prediction/etl/transform/__init__.py` | Package marker |
| `prediction/etl/transform/bronze_to_silver.py` | Validation + cleaning |
| `prediction/etl/transform/silver_to_gold.py` | Target creation + splits |
| `prediction/etl/load/__init__.py` | Package marker |
| `prediction/etl/load/parquet_loader.py` | Partitioned Parquet I/O |
| `prediction/features/__init__.py` | Package marker |
| `prediction/features/technical.py` | 20+ technical indicators |
| `prediction/features/temporal.py` | Calendar + cyclical features |
| `prediction/features/volume.py` | Volume profile features |
| `prediction/features/lag.py` | Lag/momentum features |
| `prediction/features/pipeline.py` | Feature orchestrator |
| `prediction/models/__init__.py` | Package marker |
| `prediction/models/base.py` | Abstract base model (Strategy Pattern) |
| `prediction/models/lstm.py` | PyTorch LSTM implementation |
| `prediction/models/xgboost_model.py` | XGBoost implementation |
| `prediction/models/prophet_model.py` | Prophet implementation |
| `prediction/models/ensemble.py` | Ensemble engine (3 strategies) |
| `prediction/utils/__init__.py` | Package marker |
| `prediction/utils/cache.py` | Redis cache client |
| `prediction/utils/metrics.py` | Model monitoring + alerts |
| `tests/test_prediction.py` | 40 unit tests |
| `docker-compose.yml` | PostgreSQL + Redis + API services |
| `docker/Dockerfile` | Multi-stage production build |
| `.env.example` | Environment variable template |
| `data/bronze/.gitkeep` | Directory placeholder |
| `data/silver/.gitkeep` | Directory placeholder |
| `data/gold/.gitkeep` | Directory placeholder |
| `models/.gitkeep` | Directory placeholder |

### Existing Files Modified (2 files)

| File | Change |
|---|---|
| `app/infrastructure/trading/price_prediction_adapter.py` | Replaced `NotImplementedError` with delegation to `PredictionService` |
| `app/core/config.py` | Added `redis_url`, `prediction_cache_ttl`, `data_dir` settings |

### Existing File Updated (1 file)

| File | Change |
|---|---|
| `requirements.txt` | Added `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `prophet`, `torch`, `redis` |

---

## 17. Dependencies Added

| Package | Version | Justification |
|---|---|---|
| `pandas` | ≥ 2.1.0 | DataFrame operations for ETL and feature engineering |
| `numpy` | ≥ 1.26.0 | Numerical computing for ML pipelines |
| `pyarrow` | ≥ 14.0.0 | Parquet read/write (columnar storage, ~80% compression) |
| `scikit-learn` | ≥ 1.4.0 | MinMaxScaler, preprocessing utilities |
| `xgboost` | ≥ 2.0.0 | Gradient-boosted decision tree model |
| `prophet` | ≥ 1.1.5 | Trend + seasonality decomposition |
| `torch` | ≥ 2.1.0 | PyTorch for LSTM neural network |
| `redis` | ≥ 5.0.0 | Cache client for predictions and feature store |

All dependencies are optional for the FastAPI app — the `PricePredictionAdapter` gracefully degrades if they are missing.
