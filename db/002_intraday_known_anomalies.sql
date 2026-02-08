-- ============================================================
-- FixTrade Migration 002: Intraday Ticks + Known Anomalies
-- PostgreSQL 17 — BVMT Stock Prediction System
-- ============================================================

-- ── Intraday Tick Data (1-min bars & raw tick-by-tick) ───────
CREATE TABLE IF NOT EXISTS intraday_ticks (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(50)    NOT NULL,
    tick_timestamp  TIMESTAMPTZ    NOT NULL,
    price           NUMERIC(12,3)  NOT NULL,
    volume          BIGINT         DEFAULT 0,
    tick_type       VARCHAR(10)    NOT NULL DEFAULT '1min'
                        CHECK (tick_type IN ('1min', 'tick')),
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_intraday_symbol_ts_type UNIQUE (symbol, tick_timestamp, tick_type)
);

CREATE INDEX IF NOT EXISTS idx_intraday_symbol
    ON intraday_ticks (symbol);
CREATE INDEX IF NOT EXISTS idx_intraday_ts
    ON intraday_ticks (tick_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol_ts
    ON intraday_ticks (symbol, tick_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_intraday_type
    ON intraday_ticks (tick_type);

-- ── Known / Labeled Anomalies (ground-truth for evaluation) ──
CREATE TABLE IF NOT EXISTS known_anomalies (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(50)    NOT NULL,
    anomaly_date    DATE           NOT NULL,
    anomaly_type    VARCHAR(50)    NOT NULL,
    severity        NUMERIC(5,4)   DEFAULT 0.5,
    description     TEXT,
    source          VARCHAR(50)    DEFAULT 'manual',
    verified        BOOLEAN        DEFAULT TRUE,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_known_anomaly UNIQUE (symbol, anomaly_date, anomaly_type)
);

CREATE INDEX IF NOT EXISTS idx_known_anomalies_symbol
    ON known_anomalies (symbol);
CREATE INDEX IF NOT EXISTS idx_known_anomalies_date
    ON known_anomalies (anomaly_date DESC);

-- ── Verify ───────────────────────────────────────────────────
SELECT 'Migration 002 applied successfully' AS status;
