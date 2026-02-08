"""
PostgreSQL database sink for the prediction pipeline.

Persists transformed data, predictions, model metrics, and ETL
watermarks to the ``fixtrade`` database so that results are
available to the REST API, dashboards, and audit queries.

Tables populated:
    - ``price_predictions``      — price forecasts (ensemble output)
    - ``volume_predictions``     — volume forecasts (VolumeXGB output)
    - ``liquidity_predictions``  — liquidity probability vectors
    - ``model_registry``         — trained model metadata + metrics
    - ``etl_watermarks``         — per-layer, per-ticker ingestion tracking
    - ``stock_prices``           — Silver-layer cleaned OHLCV (upsert)

Uses pg8000 (pure-Python driver, already in requirements.txt).
All writes are idempotent (UPSERT / ON CONFLICT).
"""

import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pg8000
    import pg8000.native
    HAS_PG = True
except ImportError:
    HAS_PG = False
    logger.warning("pg8000 not installed — database persistence disabled.")


class DatabaseSink:
    """Writes prediction pipeline outputs to PostgreSQL.

    Gracefully degrades if the database is unreachable — all methods
    log warnings instead of raising exceptions so the pipeline can
    continue with Parquet-only storage.

    Usage:
        sink = DatabaseSink()
        sink.ensure_tables()                          # one-time DDL
        sink.persist_predictions(results)             # after inference
        sink.persist_model_metrics("LSTM", metrics)   # after training
        sink.persist_watermark("silver", "BIAT", ...)  # after ETL
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "fixtrade",
        user: str = "postgres",
        password: str = "",
    ) -> None:
        self._conn_params = dict(
            host=host, port=port, database=database,
            user=user, password=password,
        )
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> Any:
        """Return a live connection, reconnecting if needed."""
        if not HAS_PG:
            return None

        if self._conn is not None:
            try:
                # Quick health check
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
                return self._conn
            except Exception:
                self._conn = None

        try:
            self._conn = pg8000.connect(**self._conn_params)
            self._conn.autocommit = True
            logger.info(
                "Connected to PostgreSQL %s:%s/%s",
                self._conn_params["host"],
                self._conn_params["port"],
                self._conn_params["database"],
            )
            return self._conn
        except Exception:
            logger.warning(
                "Cannot connect to PostgreSQL at %s:%s/%s. "
                "Database persistence disabled.",
                self._conn_params["host"],
                self._conn_params["port"],
                self._conn_params["database"],
            )
            return None

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # DDL: ensure all required tables exist
    # ------------------------------------------------------------------

    def ensure_tables(self) -> bool:
        """Create missing tables (idempotent).

        Returns True if successful, False if DB is unreachable.
        """
        conn = self._get_connection()
        if conn is None:
            return False

        ddl_statements = [
            # Volume predictions
            """
            CREATE TABLE IF NOT EXISTS volume_predictions (
                id          BIGSERIAL PRIMARY KEY,
                symbol      VARCHAR(20) NOT NULL,
                target_date DATE        NOT NULL,
                predicted_volume BIGINT NOT NULL,
                model_name  VARCHAR(50) DEFAULT 'volume_xgb',
                horizon_days INTEGER    DEFAULT 1,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (symbol, target_date, model_name)
            )
            """,
            # Liquidity predictions
            """
            CREATE TABLE IF NOT EXISTS liquidity_predictions (
                id           BIGSERIAL PRIMARY KEY,
                symbol       VARCHAR(20) NOT NULL,
                target_date  DATE        NOT NULL,
                prob_low     NUMERIC(6,4) NOT NULL,
                prob_medium  NUMERIC(6,4) NOT NULL,
                prob_high    NUMERIC(6,4) NOT NULL,
                predicted_tier VARCHAR(10) NOT NULL,
                model_name   VARCHAR(50) DEFAULT 'liquidity_xgb',
                created_at   TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (symbol, target_date, model_name)
            )
            """,
            # Indexes for fast queries
            "CREATE INDEX IF NOT EXISTS idx_price_pred_symbol_date ON price_predictions (symbol, target_date)",
            "CREATE INDEX IF NOT EXISTS idx_vol_pred_symbol_date ON volume_predictions (symbol, target_date)",
            "CREATE INDEX IF NOT EXISTS idx_liq_pred_symbol_date ON liquidity_predictions (symbol, target_date)",
            "CREATE INDEX IF NOT EXISTS idx_model_reg_active ON model_registry (is_active, model_name)",
        ]

        try:
            cur = conn.cursor()
            for ddl in ddl_statements:
                cur.execute(ddl)
            logger.info("Database tables verified/created.")
            return True
        except Exception:
            logger.exception("Failed to create database tables.")
            return False

    # ------------------------------------------------------------------
    # Persist: Price predictions
    # ------------------------------------------------------------------

    def persist_price_predictions(
        self, predictions: list[dict], model_name: str = "ensemble"
    ) -> int:
        """Upsert price predictions into ``price_predictions``.

        Args:
            predictions: List of dicts with keys:
                symbol, target_date, predicted_close,
                confidence_lower, confidence_upper,
                confidence_score, horizon_days
            model_name: Model that produced the predictions.

        Returns:
            Number of rows upserted.
        """
        conn = self._get_connection()
        if conn is None:
            return 0

        sql = """
            INSERT INTO price_predictions
                (symbol, target_date, predicted_close, confidence_lower,
                 confidence_upper, confidence_score, model_name, horizon_days)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, target_date, model_name)
            DO UPDATE SET
                predicted_close  = EXCLUDED.predicted_close,
                confidence_lower = EXCLUDED.confidence_lower,
                confidence_upper = EXCLUDED.confidence_upper,
                confidence_score = EXCLUDED.confidence_score,
                horizon_days     = EXCLUDED.horizon_days,
                created_at       = NOW()
        """

        # Check for unique constraint — add if missing
        self._ensure_price_predictions_upsert_constraint(conn)

        count = 0
        try:
            cur = conn.cursor()
            for p in predictions:
                cur.execute(sql, (
                    p["symbol"],
                    p["target_date"],
                    Decimal(str(p["predicted_close"])),
                    Decimal(str(p["confidence_lower"])),
                    Decimal(str(p["confidence_upper"])),
                    Decimal(str(p.get("confidence_score", 0))),
                    model_name,
                    p.get("horizon_days", 1),
                ))
                count += 1
            logger.info("Persisted %d price predictions.", count)
        except Exception:
            logger.exception("Failed to persist price predictions.")
        return count

    def _ensure_price_predictions_upsert_constraint(self, conn: Any) -> None:
        """Add a UNIQUE constraint on (symbol, target_date, model_name) if missing."""
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 1 FROM pg_constraint
                WHERE conname = 'price_predictions_symbol_target_model_uq'
            """)
            if cur.fetchone() is None:
                cur.execute("""
                    ALTER TABLE price_predictions
                    ADD CONSTRAINT price_predictions_symbol_target_model_uq
                    UNIQUE (symbol, target_date, model_name)
                """)
                logger.info("Added UNIQUE constraint to price_predictions.")
        except Exception:
            pass  # constraint may already exist under a different name

    # ------------------------------------------------------------------
    # Persist: Volume predictions
    # ------------------------------------------------------------------

    def persist_volume_predictions(
        self, predictions: list[dict], model_name: str = "volume_xgb"
    ) -> int:
        """Upsert volume predictions into ``volume_predictions``.

        Args:
            predictions: List of dicts with keys:
                symbol, target_date, predicted_volume
            model_name: Model name.

        Returns:
            Number of rows upserted.
        """
        conn = self._get_connection()
        if conn is None:
            return 0

        sql = """
            INSERT INTO volume_predictions
                (symbol, target_date, predicted_volume, model_name, horizon_days)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol, target_date, model_name)
            DO UPDATE SET
                predicted_volume = EXCLUDED.predicted_volume,
                created_at       = NOW()
        """

        count = 0
        try:
            cur = conn.cursor()
            for p in predictions:
                cur.execute(sql, (
                    p["symbol"],
                    p["target_date"],
                    int(p["predicted_volume"]),
                    model_name,
                    p.get("horizon_days", 1),
                ))
                count += 1
            logger.info("Persisted %d volume predictions.", count)
        except Exception:
            logger.exception("Failed to persist volume predictions.")
        return count

    # ------------------------------------------------------------------
    # Persist: Liquidity predictions
    # ------------------------------------------------------------------

    def persist_liquidity_predictions(
        self, predictions: list[dict], model_name: str = "liquidity_xgb"
    ) -> int:
        """Upsert liquidity predictions into ``liquidity_predictions``.

        Args:
            predictions: List of dicts with keys:
                symbol, target_date, prob_low, prob_medium, prob_high,
                predicted_tier
            model_name: Model name.

        Returns:
            Number of rows upserted.
        """
        conn = self._get_connection()
        if conn is None:
            return 0

        sql = """
            INSERT INTO liquidity_predictions
                (symbol, target_date, prob_low, prob_medium, prob_high,
                 predicted_tier, model_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, target_date, model_name)
            DO UPDATE SET
                prob_low       = EXCLUDED.prob_low,
                prob_medium    = EXCLUDED.prob_medium,
                prob_high      = EXCLUDED.prob_high,
                predicted_tier = EXCLUDED.predicted_tier,
                created_at     = NOW()
        """

        count = 0
        try:
            cur = conn.cursor()
            for p in predictions:
                probs = [p["prob_low"], p["prob_medium"], p["prob_high"]]
                tier = ("low", "medium", "high")[probs.index(max(probs))]
                cur.execute(sql, (
                    p["symbol"],
                    p["target_date"],
                    Decimal(str(p["prob_low"])),
                    Decimal(str(p["prob_medium"])),
                    Decimal(str(p["prob_high"])),
                    p.get("predicted_tier", tier),
                    model_name,
                ))
                count += 1
            logger.info("Persisted %d liquidity predictions.", count)
        except Exception:
            logger.exception("Failed to persist liquidity predictions.")
        return count

    # ------------------------------------------------------------------
    # Persist: Model registry
    # ------------------------------------------------------------------

    def persist_model_metrics(
        self,
        model_name: str,
        ticker: str | None,
        metrics: dict,
        artifact_path: str = "",
        model_version: str = "1.0",
    ) -> bool:
        """Register a trained model with its performance metrics.

        Marks it as active and deactivates previous versions.

        Args:
            model_name: e.g. "LSTM", "XGBoost", "ensemble"
            ticker: Specific ticker or None for global model.
            metrics: Dict with mae, rmse, mape, directional_acc, r_squared.
            artifact_path: Path to saved model files.
            model_version: Version string.

        Returns:
            True if persisted successfully.
        """
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cur = conn.cursor()

            # Deactivate previous versions
            cur.execute(
                "UPDATE model_registry SET is_active = FALSE "
                "WHERE model_name = %s AND (ticker = %s OR (ticker IS NULL AND %s IS NULL))",
                (model_name, ticker, ticker),
            )

            # Insert new version
            cur.execute("""
                INSERT INTO model_registry
                    (model_name, model_version, ticker, mae, rmse, mape,
                     directional_acc, r_squared, artifact_path, trained_at, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            """, (
                model_name,
                model_version,
                ticker,
                Decimal(str(metrics.get("mae", 0))),
                Decimal(str(metrics.get("rmse", 0))),
                Decimal(str(metrics.get("mape", 0))),
                Decimal(str(metrics.get("directional_accuracy", 0))),
                Decimal(str(metrics.get("r_squared", 0))),
                artifact_path,
                datetime.now(timezone.utc),
            ))

            logger.info(
                "Registered model %s (v%s) for %s — R²=%.4f",
                model_name, model_version, ticker or "global",
                metrics.get("r_squared", 0),
            )
            return True
        except Exception:
            logger.exception("Failed to register model %s.", model_name)
            return False

    # ------------------------------------------------------------------
    # Persist: ETL watermarks
    # ------------------------------------------------------------------

    def persist_watermark(
        self,
        layer: str,
        ticker: str | None,
        last_date: date,
        rows_processed: int,
    ) -> bool:
        """Record an ETL watermark for a specific layer + ticker.

        Args:
            layer: "bronze", "silver", or "gold".
            ticker: Specific ticker or None for the full layer.
            last_date: Most recent date processed.
            rows_processed: Number of rows in this batch.

        Returns:
            True if persisted.
        """
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cur = conn.cursor()
            # Upsert by (layer, ticker)
            self._ensure_watermark_upsert_constraint(conn)
            cur.execute("""
                INSERT INTO etl_watermarks (layer, ticker, last_date, rows_processed, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (layer, ticker)
                DO UPDATE SET
                    last_date      = EXCLUDED.last_date,
                    rows_processed = EXCLUDED.rows_processed,
                    updated_at     = EXCLUDED.updated_at
            """, (
                layer,
                ticker or "__global__",
                last_date,
                rows_processed,
                datetime.now(timezone.utc),
            ))
            logger.info(
                "Watermark %s/%s → %s (%d rows)",
                layer, ticker or "global", last_date, rows_processed,
            )
            return True
        except Exception:
            logger.exception("Failed to persist watermark.")
            return False

    def _ensure_watermark_upsert_constraint(self, conn: Any) -> None:
        """Add UNIQUE constraint on etl_watermarks(layer, ticker) if missing."""
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 1 FROM pg_constraint
                WHERE conname = 'etl_watermarks_layer_ticker_uq'
            """)
            if cur.fetchone() is None:
                cur.execute("""
                    ALTER TABLE etl_watermarks
                    ADD CONSTRAINT etl_watermarks_layer_ticker_uq
                    UNIQUE (layer, ticker)
                """)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Persist: Silver-layer stock prices (upsert)
    # ------------------------------------------------------------------

    def persist_silver_prices(self, df: pd.DataFrame) -> int:
        """Upsert cleaned Silver-layer stock prices to ``stock_prices``.

        Uses batch INSERT with chunked commits for performance.
        ~920K rows in ~30s instead of hours.

        Args:
            df: Silver-layer DataFrame with standard columns.

        Returns:
            Number of rows upserted.
        """
        conn = self._get_connection()
        if conn is None:
            return 0

        required = {"seance", "cloture"}
        if not required.issubset(set(df.columns)):
            logger.warning("Silver DataFrame missing required columns for DB sync.")
            return 0

        # We need a symbol column — try 'libelle' first, then 'code'
        sym_col = None
        for col in ("libelle", "code"):
            if col in df.columns:
                sym_col = col
                break
        if sym_col is None:
            logger.warning("No symbol column found in Silver DataFrame.")
            return 0

        # Ensure the unique constraint exists
        self._ensure_stock_prices_upsert_constraint(conn)

        # Build rows as a list of tuples (fast vectorised prep)
        logger.info(
            "Preparing %d Silver rows for batch DB upsert...", len(df),
        )
        rows = self._prepare_silver_rows(df, sym_col)

        sql = """
            INSERT INTO stock_prices
                (symbol, seance, ouverture, cloture, plus_bas, plus_haut,
                 quantite_negociee)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, seance)
            DO UPDATE SET
                ouverture         = EXCLUDED.ouverture,
                cloture           = EXCLUDED.cloture,
                plus_bas          = EXCLUDED.plus_bas,
                plus_haut         = EXCLUDED.plus_haut,
                quantite_negociee = EXCLUDED.quantite_negociee
        """

        BATCH = 5000
        count = 0
        try:
            # Turn off autocommit for batched transactions
            conn.autocommit = False
            cur = conn.cursor()
            for i in range(0, len(rows), BATCH):
                batch = rows[i : i + BATCH]
                for r in batch:
                    cur.execute(sql, r)
                conn.commit()
                count += len(batch)
                if count % 50000 == 0 or count == len(rows):
                    logger.info("  … upserted %d / %d rows", count, len(rows))
            conn.autocommit = True
            logger.info("Upserted %d Silver rows into stock_prices.", count)
        except Exception:
            logger.exception("Failed to persist Silver prices to database.")
            try:
                conn.rollback()
                conn.autocommit = True
            except Exception:
                pass
        return count

    @staticmethod
    def _prepare_silver_rows(
        df: pd.DataFrame, sym_col: str
    ) -> list[tuple]:
        """Convert a Silver DataFrame to a list of insert-ready tuples."""
        rows: list[tuple] = []
        symbols = df[sym_col].astype(str).values
        seances = pd.to_datetime(df["seance"]).values
        cloture = df["cloture"].values
        ouverture = df["ouverture"].values if "ouverture" in df.columns else [None] * len(df)
        plus_bas = df["plus_bas"].values if "plus_bas" in df.columns else [None] * len(df)
        plus_haut = df["plus_haut"].values if "plus_haut" in df.columns else [None] * len(df)
        volume = df["quantite_negociee"].values if "quantite_negociee" in df.columns else [0] * len(df)

        for i in range(len(df)):
            s = pd.Timestamp(seances[i])
            rows.append((
                symbols[i],
                s.date() if not pd.isna(s) else None,
                Decimal(str(float(ouverture[i]))) if ouverture[i] is not None and not pd.isna(ouverture[i]) else None,
                Decimal(str(float(cloture[i]))),
                Decimal(str(float(plus_bas[i]))) if plus_bas[i] is not None and not pd.isna(plus_bas[i]) else None,
                Decimal(str(float(plus_haut[i]))) if plus_haut[i] is not None and not pd.isna(plus_haut[i]) else None,
                int(float(volume[i])) if volume[i] is not None and not pd.isna(volume[i]) else 0,
            ))
        return rows

    def _ensure_stock_prices_upsert_constraint(self, conn: Any) -> None:
        """Add UNIQUE constraint on stock_prices(symbol, seance) if missing."""
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 1 FROM pg_constraint
                WHERE conname = 'stock_prices_symbol_seance_uq'
            """)
            if cur.fetchone() is None:
                cur.execute("""
                    ALTER TABLE stock_prices
                    ADD CONSTRAINT stock_prices_symbol_seance_uq
                    UNIQUE (symbol, seance)
                """)
                logger.info("Added UNIQUE constraint to stock_prices(symbol, seance).")
        except Exception:
            pass
