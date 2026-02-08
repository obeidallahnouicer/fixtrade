-- ============================================================
-- FixTrade Database Schema
-- PostgreSQL 17 — BVMT Stock Prediction System
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Raw Market Data (Bronze equivalent in DB) ───────────────
CREATE TABLE IF NOT EXISTS stock_prices (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(50)    NOT NULL,
    code_isin       VARCHAR(30),
    groupe          VARCHAR(50),
    seance          DATE           NOT NULL,
    ouverture       NUMERIC(12,3),
    cloture         NUMERIC(12,3)  NOT NULL,
    plus_bas        NUMERIC(12,3),
    plus_haut       NUMERIC(12,3),
    quantite_negociee BIGINT       DEFAULT 0,
    nb_transaction  INTEGER        DEFAULT 0,
    capitaux        NUMERIC(18,3)  DEFAULT 0,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_stock_prices_symbol_date UNIQUE (symbol, seance)
);

CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_stock_prices_seance ON stock_prices (seance);
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_seance ON stock_prices (symbol, seance DESC);

-- ── Price Predictions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS price_predictions (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    target_date     DATE           NOT NULL,
    predicted_close NUMERIC(12,3)  NOT NULL,
    confidence_lower NUMERIC(12,3),
    confidence_upper NUMERIC(12,3),
    confidence_score NUMERIC(5,4),
    model_name      VARCHAR(50)    DEFAULT 'ensemble',
    horizon_days    INTEGER        DEFAULT 1,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_predictions_symbol_date_model UNIQUE (symbol, target_date, model_name)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON price_predictions (symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON price_predictions (created_at DESC);

-- ── Scraped Articles ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scraped_articles (
    id              BIGSERIAL PRIMARY KEY,
    url             VARCHAR(1024)  NOT NULL,
    title           VARCHAR(512),
    summary         TEXT,
    content         TEXT,
    published_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uix_url UNIQUE (url)
);

CREATE INDEX IF NOT EXISTS idx_scraped_articles_published
    ON scraped_articles (published_at DESC);

-- ── Article Sentiments (per-article NLP result) ──────────────
CREATE TABLE IF NOT EXISTS article_sentiments (
    id              BIGSERIAL PRIMARY KEY,
    article_id      BIGINT         NOT NULL REFERENCES scraped_articles(id) ON DELETE CASCADE,
    sentiment_label VARCHAR(10)    NOT NULL CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    sentiment_score INTEGER        NOT NULL CHECK (sentiment_score IN (-1, 0, 1)),
    confidence      NUMERIC(5,4),
    analyzed_at     TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_article_sentiment UNIQUE (article_id)
);

CREATE INDEX IF NOT EXISTS idx_article_sentiments_article
    ON article_sentiments (article_id);
CREATE INDEX IF NOT EXISTS idx_article_sentiments_analyzed
    ON article_sentiments (analyzed_at DESC);

-- ── Sentiment Scores (aggregated daily per symbol) ───────────
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    score_date      DATE           NOT NULL,
    score           NUMERIC(5,4)   NOT NULL,
    sentiment       VARCHAR(10)    NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    article_count   INTEGER        DEFAULT 0,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_sentiment_symbol_date UNIQUE (symbol, score_date)
);

-- ── Anomaly Alerts ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id              UUID           PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20)    NOT NULL,
    detected_at     TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    anomaly_type    VARCHAR(50)    NOT NULL,
    severity        NUMERIC(5,4)   NOT NULL,
    description     TEXT,
    resolved        BOOLEAN        DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_anomalies_symbol ON anomaly_alerts (symbol);
CREATE INDEX IF NOT EXISTS idx_anomalies_detected ON anomaly_alerts (detected_at DESC);

-- ── Portfolios ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolios (
    id              UUID           PRIMARY KEY DEFAULT uuid_generate_v4(),
    risk_profile    VARCHAR(20)    NOT NULL DEFAULT 'moderate'
                        CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    initial_capital NUMERIC(15,2)  NOT NULL DEFAULT 10000.00,
    cash_balance    NUMERIC(15,2)  NOT NULL DEFAULT 10000.00,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    updated_at      TIMESTAMPTZ    DEFAULT NOW()
);

-- ── Portfolio Positions ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id              BIGSERIAL PRIMARY KEY,
    portfolio_id    UUID           NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol          VARCHAR(20)    NOT NULL,
    quantity        INTEGER        NOT NULL CHECK (quantity > 0),
    purchase_price  NUMERIC(12,3)  NOT NULL,
    purchased_at    DATE           NOT NULL,
    sold_at         DATE,
    sell_price      NUMERIC(12,3),

    CONSTRAINT uq_position UNIQUE (portfolio_id, symbol, purchased_at)
);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON portfolio_positions (portfolio_id);

-- ── Trade Recommendations ────────────────────────────────────
CREATE TABLE IF NOT EXISTS trade_recommendations (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    action          VARCHAR(10)    NOT NULL CHECK (action IN ('buy', 'sell', 'hold')),
    confidence      NUMERIC(5,4)   NOT NULL,
    reasoning       TEXT,
    created_at      TIMESTAMPTZ    DEFAULT NOW()
);

-- ── Model Registry (track trained models) ────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              BIGSERIAL PRIMARY KEY,
    model_name      VARCHAR(50)    NOT NULL,
    model_version   VARCHAR(20)    NOT NULL,
    ticker          VARCHAR(20),
    mae             NUMERIC(12,6),
    rmse            NUMERIC(12,6),
    mape            NUMERIC(8,4),
    directional_acc NUMERIC(5,4),
    r_squared       NUMERIC(8,6),
    artifact_path   TEXT,
    trained_at      TIMESTAMPTZ    DEFAULT NOW(),
    is_active       BOOLEAN        DEFAULT TRUE,

    CONSTRAINT uq_model_version UNIQUE (model_name, model_version, ticker)
);

-- ── ETL Watermarks (track incremental loads) ─────────────────
CREATE TABLE IF NOT EXISTS etl_watermarks (
    id              BIGSERIAL PRIMARY KEY,
    layer           VARCHAR(10)    NOT NULL CHECK (layer IN ('bronze', 'silver', 'gold')),
    ticker          VARCHAR(20),
    last_date       DATE           NOT NULL,
    rows_processed  BIGINT         DEFAULT 0,
    updated_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_watermark UNIQUE (layer, ticker)
);

-- ── Useful Views ─────────────────────────────────────────────

-- Latest price per stock
CREATE OR REPLACE VIEW v_latest_prices AS
SELECT DISTINCT ON (symbol)
    symbol, seance, ouverture, cloture, plus_bas, plus_haut,
    quantite_negociee, nb_transaction, capitaux
FROM stock_prices
ORDER BY symbol, seance DESC;

-- Latest prediction per stock
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (symbol)
    symbol, target_date, predicted_close,
    confidence_lower, confidence_upper,
    confidence_score, model_name, created_at
FROM price_predictions
ORDER BY symbol, created_at DESC;

-- Daily portfolio value (join positions with latest prices)
CREATE OR REPLACE VIEW v_portfolio_value AS
SELECT
    p.id AS portfolio_id,
    p.cash_balance,
    COALESCE(SUM(pp.quantity * lp.cloture), 0) AS positions_value,
    p.cash_balance + COALESCE(SUM(pp.quantity * lp.cloture), 0) AS total_value,
    p.initial_capital,
    ROUND(
        ((p.cash_balance + COALESCE(SUM(pp.quantity * lp.cloture), 0) - p.initial_capital)
        / p.initial_capital * 100)::NUMERIC, 2
    ) AS pnl_pct
FROM portfolios p
LEFT JOIN portfolio_positions pp ON pp.portfolio_id = p.id AND pp.sold_at IS NULL
LEFT JOIN v_latest_prices lp ON lp.symbol = pp.symbol
GROUP BY p.id, p.cash_balance, p.initial_capital;

-- ── Done ─────────────────────────────────────────────────────
-- Verify creation
SELECT 'Schema created successfully' AS status;
