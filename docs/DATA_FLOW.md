# FixTrade — Data-Directed Prediction Guide

> Follow a single row of data from a raw CSV file all the way to a price prediction.
> Every transformation is shown with **what the data looks like** at each step.

---

## Overview: The Journey of a Data Point

```
┌─────────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌────────────┐
│  RAW CSV    │──▶│ BRONZE  │──▶│  SILVER  │──▶│ FEATURES │──▶│    GOLD    │──▶│ PREDICTION │
│  (data/raw) │   │ (as-is) │   │ (clean)  │   │ (50+ cols)│  │(train/val/ │   │ (1-5 days) │
│             │   │         │   │          │   │           │  │ test+target)│  │            │
└─────────────┘   └─────────┘   └──────────┘   └──────────┘   └────────────┘   └────────────┘
       │                │              │              │               │               │
    Step 1           Step 2         Step 3         Step 4          Step 5          Step 6
```

---

## Step 1 — Raw CSV Files → Extract

**Where**: `data/raw/*.csv`
**Code**: `prediction/etl/extract/bvmt_extractor.py` → `BVMTExtractor`

### What the raw data looks like

Your CSV files contain daily BVMT trade data. Columns may vary by source:

| SEANCE     | CODE | LIBELLE | OUVERTURE | CLOTURE | PLUS_HAUT | PLUS_BAS | QUANTITE_NEGOCIEE | VOLUME    | VARIATION |
|------------|------|---------|-----------|---------|-----------|----------|-------------------|-----------|-----------|
| 2024-06-03 | BIAT | BIAT    | 96.50     | 97.00   | 97.50     | 96.20    | 12345             | 1197345.0 | +0.52%    |
| 2024-06-03 | SFBT | SFBT    | 18.40     | 18.50   | 18.60     | 18.30    | 8901              | 164768.5  | +0.54%    |

### What the extractor does

1. **Reads every CSV** in `data/raw/` — tries UTF-8, Latin-1, CP1252 encodings and `,` `;` `\t` separators
2. **Renames columns** to a standard schema: `SEANCE` → `seance`, `CLOTURE` → `cloture`, etc.
3. **Deduplicates** on `(seance, code)` — keeps first occurrence
4. **Saves to Bronze** as partitioned Parquet: `data/bronze/year=2024/month=06/code=BIAT/data.parquet`

### Data shape after Extract

| seance     | code | libelle | ouverture | cloture | plus_haut | plus_bas | quantite_negociee | volume    | variation |
|------------|------|---------|-----------|---------|-----------|----------|-------------------|-----------|-----------|
| 2024-06-03 | BIAT | BIAT    | 96.50     | 97.00   | 97.50     | 96.20    | 12345             | 1197345.0 | +0.52%    |

> **Key point**: Bronze = raw data, just with consistent naming. Nothing is changed.

---

## Step 2 — Bronze → Silver (Validation & Cleaning)

**Code**: `prediction/etl/transform/bronze_to_silver.py` → `BronzeToSilverTransformer`

### 4 Validation Rules

Every row must pass ALL of these:

| # | Rule Name              | Logic                                 | What it catches            |
|---|------------------------|---------------------------------------|----------------------------|
| 1 | `cloture_positive`     | `cloture > 0`                         | Erroneous zero/negative closes |
| 2 | `high_gte_low`         | `plus_haut >= plus_bas`               | Swapped high/low values    |
| 3 | `volume_non_negative`  | `quantite_negociee >= 0`              | Negative volume entries    |
| 4 | `no_future_dates`      | `seance <= today`                     | Future-dated rows          |

Rows that fail → rejected and logged. Rows that pass → continue.

### Cleaning steps

1. **Type coercion**: `seance` → datetime, price columns → float64, `quantite_negociee` → int
2. **Null handling**: Forward-fill price NaNs within each ticker group (ffill), then drop rows still missing `cloture`
3. **Sort**: by `(code, seance)` ascending

### Data shape after Silver

| seance (datetime64) | code (str) | libelle | ouverture (float64) | cloture (float64) | plus_haut (float64) | plus_bas (float64) | quantite_negociee (int64) |
|----------------------|-----------|---------|---------------------|--------------------|---------------------|--------------------|---------------------------|
| 2024-06-03           | BIAT      | BIAT    | 96.50               | 97.00              | 97.50               | 96.20              | 12345                     |

> **Key point**: Silver = validated, typed, no nulls in critical columns. This is the "source of truth" for features.

---

## Step 3 — Silver → Features (50+ Engineered Columns)

**Code**: `prediction/features/pipeline.py` → `FeaturePipeline`

The feature pipeline processes **each ticker independently** (to prevent cross-contamination), then concatenates.

### 3a — Technical Indicators (27 features)

**Code**: `prediction/features/technical.py` → `TechnicalFeatures`

All computed from `cloture`, `plus_haut`, `plus_bas`, `quantite_negociee`:

| Feature Group        | Columns Created                                    | Window Sizes      |
|----------------------|----------------------------------------------------|--------------------|
| Simple Moving Avg    | `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_200`  | 5, 10, 20, 50, 200 |
| Exponential MA       | `ema_12`, `ema_26`                                 | 12, 26             |
| RSI                  | `rsi`                                              | 14                 |
| MACD                 | `macd`, `macd_signal`, `macd_histogram`             | 12/26/9            |
| Bollinger Bands      | `bollinger_upper`, `bollinger_lower`, `bollinger_width` | 20, 2σ         |
| ATR                  | `atr`                                              | 14                 |
| Stochastic           | `stochastic_k`, `stochastic_d`                     | 14                 |
| Rate of Change       | `roc_5`, `roc_10`, `roc_20`                        | 5, 10, 20          |
| On-Balance Volume    | `obv`                                              | cumulative         |
| Price/SMA ratio      | `price_to_sma_5`, `..._10`, `..._20`, `..._50`, `..._200` | same as SMA |
| Volatility           | `volatility_20d`                                   | 20                 |

**Anti-leakage**: Every single indicator is `.shift(1)` — the value at day *t* is computed from data up to day *t−1* only.

**Example** — `sma_5` at row index 5:
```
sma_5[5] = mean(cloture[0], cloture[1], cloture[2], cloture[3], cloture[4])
         = mean of days 0..4, available at day 5 ✓
```

### 3b — Temporal / Calendar Features (16 features)

**Code**: `prediction/features/temporal.py` → `TemporalFeatures`

Computed from the `seance` date column — not shifted (they describe the current day):

| Feature                | Type    | Description                          |
|------------------------|---------|--------------------------------------|
| `day_of_week`          | int     | 0=Monday … 4=Friday                 |
| `day_of_month`         | int     | 1–31                                 |
| `month`                | int     | 1–12                                 |
| `quarter`              | int     | 1–4                                  |
| `week_of_year`         | int     | 1–52                                 |
| `day_of_year`          | int     | 1–365                                |
| `is_month_start`       | binary  | 1 if first business day of month     |
| `is_month_end`         | binary  | 1 if last business day of month      |
| `is_quarter_end`       | binary  | 1 if last day of quarter             |
| `is_year_end`          | binary  | 1 if Dec 31                          |
| `day_of_week_sin/cos`  | float   | Cyclical encoding (period=5 days)    |
| `month_sin/cos`        | float   | Cyclical encoding (period=12 months) |
| `trading_day_of_month` | int     | 1st, 2nd, … trading day per ticker   |
| `is_near_holiday`      | binary  | Within 2 days of a Tunisian holiday  |

Tunisian holidays encoded: New Year, Revolution Day (Jan 14), Independence (Mar 20), Martyrs' Day (Apr 9), Labour Day, Victory Day (Jun 1), Republic Day (Jul 25), Women's Day (Aug 13), Evacuation Day (Oct 15).

### 3c — Volume Features (8 features)

**Code**: `prediction/features/volume.py` → `VolumeFeatures`

| Feature          | Formula / Logic                                         |
|------------------|---------------------------------------------------------|
| `volume_sma_5`   | 5-day rolling mean of volume, shifted by 1              |
| `volume_sma_10`  | 10-day rolling mean of volume, shifted by 1             |
| `volume_sma_20`  | 20-day rolling mean of volume, shifted by 1             |
| `volume_ratio`   | current_vol / 20-day_avg_vol, shifted by 1              |
| `volume_trend`   | Linear slope of volume over last 5 days, shifted by 1   |
| `vwap`           | Volume-Weighted Average Price (20-day window), shifted  |
| `price_to_vwap`  | (close / vwap) − 1, shifted by 1                        |
| `ad_line`        | Accumulation/Distribution line (cumulative), shifted    |
| `mfi`            | Money Flow Index (14-day), shifted by 1                  |

### 3d — Lag & Momentum Features (15+ features)

**Code**: `prediction/features/lag.py` → `LagFeatures`

| Feature Group       | Columns                                                        |
|---------------------|----------------------------------------------------------------|
| Price lags          | `close_lag_1`, `close_lag_2`, `close_lag_3`, `close_lag_5`, `close_lag_10`, `close_lag_20` |
| Return lags         | `return_lag_1`, `return_lag_2`, …, `return_lag_20`             |
| Rolling return mean | `return_mean_5d`, `return_mean_10d`, `return_mean_20d`         |
| Rolling return std  | `return_std_5d`, `return_std_10d`, `return_std_20d`            |
| Rolling return skew | `return_skew_5d`, `return_skew_10d`, `return_skew_20d`         |
| Cumulative return   | `cum_return_5d`, `cum_return_10d`, `cum_return_20d`, `cum_return_60d` |
| Max drawdown        | `max_drawdown_20d`                                             |
| Momentum            | `momentum_5d`, `momentum_10d`, `momentum_20d`                  |
| Mean reversion      | `mean_reversion_z` (z-score vs 20-day SMA)                     |

### Data shape after Features

Your DataFrame now has **50+ columns** per row. Example for BIAT on 2024-06-03:

```
seance            : 2024-06-03
code              : BIAT
cloture           : 97.00       ← original price (still present)
sma_5             : 96.32       ← technical
rsi               : 58.7        ← technical
macd              : 0.42        ← technical
day_of_week       : 0           ← temporal (Monday)
month_sin         : 0.866       ← temporal (June, cyclical)
volume_ratio      : 1.23        ← volume (above average)
close_lag_1       : 96.50       ← lag (yesterday's close)
momentum_5d       : 0.015       ← lag (1.5% up over 5 days)
... (40+ more columns)
```

---

## Step 4 — Features → Gold (Targets + Chronological Splits)

**Code**: `prediction/etl/transform/silver_to_gold.py` → `SilverToGoldTransformer`

### Target creation

For each prediction horizon *h* ∈ {1, 2, 3, 5} days:

```
target_hd[row i] = cloture[row i + h]   (using .shift(-h) within each ticker)
```

| Row  | cloture | target_1d | target_2d | target_3d | target_5d |
|------|---------|-----------|-----------|-----------|-----------|
| i    | 97.00   | 97.50     | 97.80     | 98.10     | 98.50     |
| i+1  | 97.50   | 97.80     | 98.10     | 98.50     | 99.00     |
| …    | …       | …         | …         | …         | …         |
| last | 99.20   | NaN       | NaN       | NaN       | NaN       |

Rows with NaN targets (end of each ticker's series) → dropped from training.

### Chronological splits (no shuffling — ever)

```
──────────────────────────────────────────────────────────────────────
│       TRAIN         │     VALIDATION     │        TEST           │
│   year ≤ 2024       │   year = 2025      │   year > 2025         │
──────────────────────────────────────────────────────────────────────
      past ──────────────────────────────────────────────▶ future
```

> **Key point**: Train never sees validation data, validation never sees test data. This is how you prevent temporal leakage in financial time series.

---

## Step 5 — Gold → Model Training (Walk-Forward CV)

**Code**: `prediction/training.py` → `TrainingPipeline`

### Walk-Forward Cross-Validation (3 splits)

Unlike random CV, walk-forward preserves time order:

```
Split 1:  TRAIN [... → 2021]    VAL [2022]
Split 2:  TRAIN [... → 2022]    VAL [2023]
Split 3:  TRAIN [... → 2023]    VAL [2024]
```

For each split, **3 models are trained independently**:

### Model 1: LSTM (PyTorch)

```
Input: 60-day sequences of all features → MinMaxScaler → Stacked LSTM (2 layers, 128 hidden) → Linear → price
Training: Adam optimizer, MSE loss, gradient clipping (1.0), early stopping (patience=10)
Best for: capturing temporal patterns in high-liquidity stocks
```

### Model 2: XGBoost

```
Input: flat feature row (no sequences) → XGBRegressor (500 trees, depth=6, lr=0.05)
Training: built-in early stopping on validation set
Best for: feature interactions, medium-frequency patterns
```

### Model 3: Prophet

```
Input: date + close price + selected regressors (rsi, macd, volume_ratio, volatility_20d)
Training: automatic changepoint detection, yearly + weekly seasonality
Best for: long-term trends, seasonal patterns, low-liquidity stocks
```

### Metrics computed per model per split

| Metric                | Formula                                    | Good Value |
|-----------------------|--------------------------------------------|------------|
| MAE                   | mean(\|actual − predicted\|)               | < 1.0 TND  |
| RMSE                  | √(mean((actual − predicted)²))             | < 1.5 TND  |
| MAPE                  | mean(\|actual − predicted\| / actual)       | < 2%       |
| Directional Accuracy  | % of days where direction was correct       | > 55%      |
| R²                    | 1 − (SS_res / SS_tot)                       | > 0.90     |

After 3 CV splits → **average metrics** across splits for each model.

### Final model training

`python -m prediction train --final` trains on **all available data** and saves the ensemble to `models/`.

---

## Step 6 — Trained Models → Prediction (Inference)

**Code**: `prediction/inference.py` → `PredictionService`

### Prediction flow for a request like `predict("BIAT", days=3)`

```
                    ┌──────────────┐
                    │ Cache check  │
                    │ (Redis/mem)  │
                    └──────┬───────┘
                           │
                    hit?───┤───yes──▶ Return cached result
                           │
                    no ◀───┘
                           │
                    ┌──────▼───────┐
                    │ Load latest  │
                    │ features for │
                    │ BIAT from    │
                    │ Silver layer │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Classify     │
                    │ liquidity    │
                    │ tier         │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
        ┌───────┐    ┌──────────┐   ┌──────────┐
        │ LSTM  │    │ XGBoost  │   │ Prophet  │
        │(Tier1)│    │(Tier1+2) │   │(all tiers│
        └───┬───┘    └────┬─────┘   └────┬─────┘
            │             │              │
            └─────────────┼──────────────┘
                          ▼
                   ┌─────────────┐
                   │  Ensemble   │
                   │  Weighted   │
                   │  Average    │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Cache store │
                   │ (TTL based) │
                   └──────┬──────┘
                          │
                          ▼
                   PredictionResult
```

### Liquidity Tiers (which models are used)

| Tier   | Condition                  | Models Used             | Default Weights       |
|--------|----------------------------|-------------------------|-----------------------|
| Tier 1 | avg daily volume ≥ 10,000  | LSTM + XGBoost + Prophet | 0.45 / 0.35 / 0.20   |
| Tier 2 | volume 1,000 – 10,000      | XGBoost + Prophet        | 0.65 / 0.35          |
| Tier 3 | volume < 1,000             | Prophet only             | 1.00                  |

### Dynamic Weighting (Phase 2)

Weights shift based on market conditions:

| Market Condition        | LSTM   | XGBoost | Prophet |
|-------------------------|--------|---------|---------|
| High volatility (>3%)   | 0.20   | 0.30    | **0.50** |
| Low liquidity (<5000)   | 0.25   | 0.15    | **0.60** |
| Strong trend (<1% vol)  | **0.60** | 0.30  | 0.10    |
| Normal                  | 0.45   | 0.35    | 0.20    |

### Prediction Result

Each prediction returns:

```python
PredictionResult(
    symbol="BIAT",
    target_date=date(2024, 6, 6),        # next trading day (skips weekends)
    predicted_close=97.85,                # ensemble prediction in TND
    confidence_lower=96.20,               # lower bound (model disagreement based)
    confidence_upper=99.50,               # upper bound
    model_name="ensemble",
    confidence_score=0.72,                # 0–1 based on tier + model agreement
    horizon_days=3,
)
```

### Caching strategy

| Period              | TTL         | Reason                           |
|---------------------|-------------|----------------------------------|
| Market hours (9–15) | 1 hour      | Predictions may change intraday  |
| After market close  | 12 hours    | Stable until next session        |
| Feature cache       | 90 days     | Historical features rarely change |

Cache key patterns:
- Predictions: `pred:{TICKER}:{MODEL}` → `pred:BIAT:ensemble`
- Features: `features:{CODE}:{DATE}` → `features:BIAT:2024-06-03`

---

## The Complete Pipeline in One CLI Session

```bash
# 1. Ingest raw CSV → Bronze → Silver → Features → Gold
python -m prediction etl

# 2. Train with walk-forward CV (evaluate) or final production model (save)
python -m prediction train           # walk-forward CV → prints metrics
python -m prediction train --final   # full data → saves to models/
python -m prediction train --symbol BIAT --final — trains LSTM, XGBoost, and Prophet on just BIAT data (2502 rows, ~16 seconds), saves models to BIAT
python -m prediction predict --symbol BIAT --days 5 — loads per-symbol models, fetches features, runs ensemble inference → Close=110.365, CI=[87.938, 121.129]
# 3. Predict
python -m prediction predict --symbol BIAT --days 3

# 4. Pre-warm cache for top tickers (run daily at 8 AM)
python -m prediction warm-cache
```

---

## Monitoring & Retraining

**Code**: `prediction/utils/metrics.py` → `ModelMonitor`

The monitor tracks every evaluation and triggers alerts:

| Alert Condition                        | Action                  |
|----------------------------------------|-------------------------|
| RMSE > 1.5× threshold                 | Log warning + retrain   |
| Directional accuracy < 50%            | Log warning + retrain   |
| Drift detected in 3+ consecutive evals | Log warning + retrain   |

**Drift detection**: If `|mean(predictions) − mean(actuals)| / |mean(actuals)| > 5%`, drift is flagged.

---

## Anti-Leakage Summary

This is the most critical aspect. Here's how every component prevents data from the future leaking into training:

| Component           | Mechanism                                      | Verified by test                       |
|---------------------|------------------------------------------------|----------------------------------------|
| Technical features  | `.shift(1)` on every indicator                  | `test_all_features_shifted`            |
| Volume features     | `.shift(1)` on every indicator                  | `test_features_shifted`                |
| Lag features        | Inherently past-looking + `.shift(1)`           | `test_computes_lag_features`           |
| Temporal features   | Describe the current day (no future info)       | `test_computes_calendar_features`      |
| Gold targets        | `.shift(-h)` (forward-looking) on `cloture`     | `test_targets_are_future_looking`      |
| Train/Val/Test      | Strict chronological split, no random shuffle   | `test_creates_training_view`           |
| Walk-forward CV     | Train boundary always < Validation boundary     | Built into `TrainingPipeline`          |
| Integration test    | Feature[t] depends only on data[t-1] and earlier | `test_features_only_use_past_data`    |

---

## File Map

```
prediction/
│
├── __init__.py              ← Public API: PredictionService, ETLPipeline, TrainingPipeline
├── __main__.py              ← CLI entry point (python -m prediction)
├── config.py                ← All configuration (reads .env via app.core.config)
├── pipeline.py              ← ETLPipeline: orchestrates Extract → Bronze → Silver → Gold
├── training.py              ← TrainingPipeline: walk-forward CV + final model training
├── inference.py             ← PredictionService: cache → features → model → predict → cache
├── cli.py                   ← CLI commands: etl, train, predict, warm-cache
│
├── etl/
│   ├── extract/
│   │   └── bvmt_extractor.py   ← CSV ingestion, encoding detection, column mapping
│   ├── transform/
│   │   ├── bronze_to_silver.py  ← Validation (4 rules), type coercion, null handling
│   │   └── silver_to_gold.py    ← Target creation, chronological splits
│   └── load/
│       └── parquet_loader.py    ← Partitioned Parquet I/O, watermark tracking
│
├── features/
│   ├── pipeline.py          ← FeaturePipeline: orchestrates all 4 generators
│   ├── technical.py         ← 27 indicators (SMA, RSI, MACD, Bollinger, ATR, …)
│   ├── temporal.py          ← 16 calendar features + Tunisian holidays
│   ├── volume.py            ← 8 volume-based features (VWAP, MFI, A/D line)
│   └── lag.py               ← 15+ lag/momentum features (returns, drawdown, z-score)
│
├── models/
│   ├── base.py              ← BasePredictionModel (Strategy Pattern ABC)
│   ├── lstm.py              ← LSTMPredictor (PyTorch stacked LSTM)
│   ├── xgboost_model.py     ← XGBoostPredictor (gradient-boosted trees)
│   ├── prophet_model.py     ← ProphetPredictor (trend + seasonality)
│   └── ensemble.py          ← EnsemblePredictor (weighted avg + liquidity tiers)
│
└── utils/
    ├── cache.py             ← CacheClient (Redis + in-memory fallback)
    └── metrics.py           ← ModelMonitor (drift detection, retrain alerts)
```

---

## Configuration at a Glance

All infrastructure settings come from `.env` via `app.core.config.settings`.
ML hyperparameters are in `prediction/config.py` as frozen dataclasses.

| Config Class       | What it controls                          | Source       |
|--------------------|-------------------------------------------|--------------|
| `PathsConfig`      | data/bronze, data/silver, data/gold, models/ | `.env` (FIXTRADE_DATA_DIR, MODEL_DIR) |
| `DatabaseConfig`   | PostgreSQL host, port, db, user, password | `.env` (POSTGRES_*) |
| `RedisConfig`      | Redis host, port, TTLs                    | `.env` (REDIS_*) |
| `ModelConfig`      | LSTM/XGBoost/Prophet hyperparameters      | Hardcoded (tuned by experimentation) |
| `FeatureConfig`    | Window sizes, lag days, horizons          | Hardcoded (tuned by experimentation) |
| `LiquidityTierConfig` | Volume thresholds for tier 1/2/3       | Hardcoded    |

---

## Test Coverage

60 tests total — all passing ✅

| Test Class              | Count | What it validates                                    |
|-------------------------|-------|------------------------------------------------------|
| `TestConfig`            | 3     | Config loads, paths consistent, weights sum to 1     |
| `TestTechnicalFeatures` | 5     | SMA, RSI, MACD compute correctly + shift(1) verified |
| `TestTemporalFeatures`  | 2     | Calendar features + cyclical encoding bounds          |
| `TestVolumeFeatures`    | 2     | Volume indicators + shift verification                |
| `TestLagFeatures`       | 1     | Lag/momentum columns present                          |
| `TestFeaturePipeline`   | 1     | Full pipeline produces 30+ feature columns            |
| `TestDataQuality`       | 2     | Positive close, high ≥ low validation                 |
| `TestBronzeToSilver`    | 2     | Transform types correct + sorted                      |
| `TestSilverToGold`      | 3     | Training view, targets, inference view                |
| `TestModelMetrics`      | 1     | MAE, RMSE, MAPE, dir_acc, R² computation              |
| `TestEnsemble`          | 4     | Tier classification, model selection, dynamic weights  |
| `TestCache`             | 3     | In-memory fallback, invalidation, feature store        |
| `TestModelMonitor`      | 3     | Evaluation, drift detection, retrain trigger           |
| `TestParquetLoader`     | 3     | Save/load roundtrip, watermark, partitioned save       |
| `TestPredictionService` | 3     | Fallback prediction, weekend skip, serialization       |
| `TestAntiLeakage`       | 2     | Features use only past data, targets are future        |
| *(app tests)*           | 20    | Domain entities, API endpoints, trading logic          |
