"""
Tests for prediction.db_sink.DatabaseSink.

Uses unittest.mock to avoid requiring a live PostgreSQL instance.
Validates SQL generation, data conversion, batch logic, graceful
degradation, and idempotent constraint creation.
"""

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pandas as pd

from prediction.db_sink import DatabaseSink


# ── Helpers ──────────────────────────────────────────────────────────


def _make_sink_with_mock_conn() -> tuple[DatabaseSink, MagicMock, MagicMock]:
    """Return a DatabaseSink wired to a mock connection + cursor."""
    sink = DatabaseSink()
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    # Health-check "SELECT 1" succeeds
    cursor.fetchone.return_value = (1,)
    sink._conn = conn
    return sink, conn, cursor


# ══════════════════════════════════════════════════════════════════════
# Connection Management
# ══════════════════════════════════════════════════════════════════════


class TestConnectionManagement(unittest.TestCase):
    """Tests for _get_connection, reconnection, and close."""

    def test_returns_none_when_pg8000_missing(self):
        with patch("prediction.db_sink.HAS_PG", False):
            sink = DatabaseSink()
            self.assertIsNone(sink._get_connection())

    def test_reuses_healthy_connection(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        result = sink._get_connection()
        self.assertIs(result, conn)
        cursor.execute.assert_called_once_with("SELECT 1")

    def test_reconnects_on_stale_connection(self):
        sink = DatabaseSink()
        stale = MagicMock()
        stale.cursor.return_value.fetchone.side_effect = Exception("closed")
        sink._conn = stale

        fresh = MagicMock()
        with patch("prediction.db_sink.pg8000.connect", return_value=fresh):
            result = sink._get_connection()
        self.assertIs(result, fresh)
        self.assertTrue(fresh.autocommit)

    def test_returns_none_on_connection_failure(self):
        sink = DatabaseSink()
        sink._conn = None
        with patch("prediction.db_sink.pg8000.connect", side_effect=Exception("refused")):
            result = sink._get_connection()
        self.assertIsNone(result)

    def test_close_calls_conn_close(self):
        sink, conn, _ = _make_sink_with_mock_conn()
        sink.close()
        conn.close.assert_called_once()
        self.assertIsNone(sink._conn)

    def test_close_is_safe_when_no_connection(self):
        sink = DatabaseSink()
        sink.close()  # should not raise


# ══════════════════════════════════════════════════════════════════════
# DDL: ensure_tables
# ══════════════════════════════════════════════════════════════════════


class TestEnsureTables(unittest.TestCase):

    def test_returns_false_when_no_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertFalse(sink.ensure_tables())

    def test_creates_tables_and_indexes(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        result = sink.ensure_tables()
        self.assertTrue(result)
        # Should execute 6 DDL statements (2 CREATE TABLE + 4 CREATE INDEX)
        executed_sql = [c.args[0] for c in cursor.execute.call_args_list
                        if c.args and isinstance(c.args[0], str)]
        # Filter out the health-check SELECT 1
        ddl = [s for s in executed_sql if "CREATE" in s.upper()]
        self.assertEqual(len(ddl), 6)

    def test_returns_false_on_ddl_failure(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        # First call is health-check, second is DDL
        cursor.execute.side_effect = [None, Exception("permission denied")]
        result = sink.ensure_tables()
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════
# Price Predictions
# ══════════════════════════════════════════════════════════════════════


class TestPersistPricePredictions(unittest.TestCase):

    def _sample_predictions(self) -> list[dict]:
        return [
            {
                "symbol": "BIAT",
                "target_date": date(2025, 7, 14),
                "predicted_close": 120.50,
                "confidence_lower": 115.0,
                "confidence_upper": 126.0,
                "confidence_score": 0.85,
                "horizon_days": 5,
            },
            {
                "symbol": "SFBT",
                "target_date": date(2025, 7, 14),
                "predicted_close": 45.30,
                "confidence_lower": 43.0,
                "confidence_upper": 47.5,
                "confidence_score": 0.72,
            },
        ]

    def test_returns_zero_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertEqual(sink.persist_price_predictions([{"symbol": "X"}]), 0)

    def test_inserts_all_rows(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        # Constraint check returns existing
        cursor.fetchone.side_effect = [(1,), (1,)]  # health-check, constraint check
        preds = self._sample_predictions()
        count = sink.persist_price_predictions(preds)
        self.assertEqual(count, 2)

    def test_converts_to_decimal(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        preds = [self._sample_predictions()[0]]
        sink.persist_price_predictions(preds)

        # Find the INSERT call (skip health-check and constraint-check calls)
        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO price_predictions" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)
        params = insert_calls[0].args[1]
        # predicted_close should be a Decimal
        self.assertIsInstance(params[2], Decimal)
        self.assertEqual(params[2], Decimal("120.5"))

    def test_handles_insert_failure_gracefully(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        # Make _ensure constraint succeed, but INSERT raises
        cursor.fetchone.return_value = (1,)

        def execute_side(sql, *args, **kwargs):
            if isinstance(sql, str) and "INSERT INTO price_predictions" in sql:
                raise Exception("duplicate key")

        cursor.execute.side_effect = execute_side
        # Should not raise — graceful degradation
        result = sink.persist_price_predictions(self._sample_predictions())
        self.assertEqual(result, 0)


# ══════════════════════════════════════════════════════════════════════
# Volume Predictions
# ══════════════════════════════════════════════════════════════════════


class TestPersistVolumePredictions(unittest.TestCase):

    def test_returns_zero_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertEqual(sink.persist_volume_predictions([]), 0)

    def test_inserts_with_correct_types(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        preds = [
            {
                "symbol": "BIAT",
                "target_date": date(2025, 7, 14),
                "predicted_volume": 12345.6,
                "horizon_days": 3,
            }
        ]
        count = sink.persist_volume_predictions(preds)
        self.assertEqual(count, 1)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO volume_predictions" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)
        params = insert_calls[0].args[1]
        self.assertEqual(params[0], "BIAT")
        self.assertEqual(params[2], 12345)  # int conversion
        self.assertEqual(params[4], 3)  # horizon_days

    def test_handles_failure_gracefully(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1 and args and "INSERT" in str(args[0]):
                raise Exception("db error")

        cursor.execute.side_effect = side_effect
        result = sink.persist_volume_predictions([{"symbol": "X", "target_date": date.today(), "predicted_volume": 1}])
        self.assertEqual(result, 0)


# ══════════════════════════════════════════════════════════════════════
# Liquidity Predictions
# ══════════════════════════════════════════════════════════════════════


class TestPersistLiquidityPredictions(unittest.TestCase):

    def test_returns_zero_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertEqual(sink.persist_liquidity_predictions([]), 0)

    def test_inserts_with_decimal_probs(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        preds = [
            {
                "symbol": "SFBT",
                "target_date": date(2025, 7, 14),
                "prob_low": 0.1,
                "prob_medium": 0.3,
                "prob_high": 0.6,
                "predicted_tier": "high",
            }
        ]
        count = sink.persist_liquidity_predictions(preds)
        self.assertEqual(count, 1)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO liquidity_predictions" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)
        params = insert_calls[0].args[1]
        self.assertIsInstance(params[2], Decimal)  # prob_low
        self.assertIsInstance(params[3], Decimal)  # prob_medium
        self.assertIsInstance(params[4], Decimal)  # prob_high
        self.assertEqual(params[5], "high")  # predicted_tier

    def test_auto_derives_tier_from_probs(self):
        """When predicted_tier is missing, it should be inferred from probs."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        preds = [
            {
                "symbol": "BIAT",
                "target_date": date(2025, 7, 14),
                "prob_low": 0.7,
                "prob_medium": 0.2,
                "prob_high": 0.1,
            }
        ]
        sink.persist_liquidity_predictions(preds)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO liquidity_predictions" in c.args[0]
        ]
        params = insert_calls[0].args[1]
        self.assertEqual(params[5], "low")  # auto-derived tier


# ══════════════════════════════════════════════════════════════════════
# Model Registry
# ══════════════════════════════════════════════════════════════════════


class TestPersistModelMetrics(unittest.TestCase):

    def test_returns_false_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertFalse(sink.persist_model_metrics("LSTM", "BIAT", {}))

    def test_deactivates_old_then_inserts_new(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        metrics = {
            "mae": 1.5,
            "rmse": 2.3,
            "mape": 0.02,
            "directional_accuracy": 0.65,
            "r_squared": 0.95,
        }
        result = sink.persist_model_metrics(
            "LSTM", "BIAT", metrics,
            artifact_path="/models/lstm_biat.pt",
            model_version="2.0",
        )
        self.assertTrue(result)

        # Expect 3 execute calls: health-check, UPDATE (deactivate), INSERT
        execute_calls = cursor.execute.call_args_list
        # Health-check
        self.assertIn("SELECT 1", execute_calls[0].args[0])
        # Deactivate old — ticker is not None so uses "ticker = %s"
        update_sql = execute_calls[1].args[0]
        self.assertIn("UPDATE model_registry SET is_active = FALSE", update_sql)
        self.assertIn("ticker = %s", update_sql)
        self.assertEqual(execute_calls[1].args[1], ("LSTM", "BIAT"))
        # Insert new
        insert_sql = execute_calls[2].args[0]
        self.assertIn("INSERT INTO model_registry", insert_sql)
        params = execute_calls[2].args[1]
        self.assertEqual(params[0], "LSTM")
        self.assertEqual(params[1], "2.0")
        self.assertEqual(params[2], "BIAT")
        self.assertEqual(params[3], Decimal("1.5"))   # mae
        self.assertEqual(params[7], Decimal("0.95"))   # r_squared
        self.assertEqual(params[8], "/models/lstm_biat.pt")

    def test_handles_global_model_with_none_ticker(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        result = sink.persist_model_metrics(
            "ensemble", None, {"r_squared": 0.99},
        )
        self.assertTrue(result)
        deactivate_call = cursor.execute.call_args_list[1]
        # ticker is None → uses "ticker IS NULL" with only model_name param
        self.assertIn("ticker IS NULL", deactivate_call.args[0])
        self.assertEqual(deactivate_call.args[1], ("ensemble",))

    def test_returns_false_on_db_error(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise Exception("db error")

        cursor.execute.side_effect = side_effect
        result = sink.persist_model_metrics("LSTM", "BIAT", {})
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════
# ETL Watermarks
# ══════════════════════════════════════════════════════════════════════


class TestPersistWatermark(unittest.TestCase):

    def test_returns_false_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertFalse(
                sink.persist_watermark("silver", "BIAT", date.today(), 100)
            )

    def test_upserts_watermark_for_ticker(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        # Constraint-check returns existing
        cursor.fetchone.side_effect = [(1,), (1,)]
        result = sink.persist_watermark("silver", "BIAT", date(2025, 12, 31), 5000)
        self.assertTrue(result)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO etl_watermarks" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)
        params = insert_calls[0].args[1]
        self.assertEqual(params[0], "silver")
        self.assertEqual(params[1], "BIAT")
        self.assertEqual(params[2], date(2025, 12, 31))
        self.assertEqual(params[3], 5000)

    def test_global_watermark_uses_dunder_global(self):
        """When ticker is None, it should store '__global__'."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        sink.persist_watermark("gold", None, date(2025, 6, 1), 100)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO etl_watermarks" in c.args[0]
        ]
        params = insert_calls[0].args[1]
        self.assertEqual(params[1], "__global__")


# ══════════════════════════════════════════════════════════════════════
# Silver Prices (batch upsert)
# ══════════════════════════════════════════════════════════════════════


class TestPersistSilverPrices(unittest.TestCase):

    def _sample_df(self, n: int = 10) -> pd.DataFrame:
        """Create a small Silver-like DataFrame."""
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "libelle": ["BIAT"] * n,
            "code": ["BIAT"] * n,
            "seance": dates,
            "ouverture": [100.0 + i for i in range(n)],
            "cloture": [101.0 + i for i in range(n)],
            "plus_bas": [99.0 + i for i in range(n)],
            "plus_haut": [102.0 + i for i in range(n)],
            "quantite_negociee": [1000 * (i + 1) for i in range(n)],
        })

    def test_returns_zero_without_connection(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertEqual(sink.persist_silver_prices(self._sample_df()), 0)

    def test_returns_zero_when_missing_required_columns(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        bad_df = pd.DataFrame({"foo": [1, 2]})
        self.assertEqual(sink.persist_silver_prices(bad_df), 0)

    def test_returns_zero_when_no_symbol_column(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        df = pd.DataFrame({
            "seance": pd.bdate_range("2025-01-01", periods=3),
            "cloture": [10, 11, 12],
        })
        self.assertEqual(sink.persist_silver_prices(df), 0)

    def test_upserts_all_rows(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]  # health + constraint
        df = self._sample_df(7)
        count = sink.persist_silver_prices(df)
        self.assertEqual(count, 7)

    def test_batch_commits(self):
        """With BATCH=5000, a 12-row frame should commit in 1 batch."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        df = self._sample_df(12)
        sink.persist_silver_prices(df)
        # commit called once per batch + autocommit setting
        commit_calls = conn.commit.call_args_list
        self.assertGreaterEqual(len(commit_calls), 1)

    def test_prefers_libelle_over_code(self):
        """Symbol column preference: libelle > code."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        df = pd.DataFrame({
            "libelle": ["SYM_L"],
            "code": ["SYM_C"],
            "seance": [pd.Timestamp("2025-01-02")],
            "cloture": [50.0],
            "ouverture": [49.0],
            "plus_bas": [48.0],
            "plus_haut": [51.0],
            "quantite_negociee": [500],
        })
        sink.persist_silver_prices(df)
        insert_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and isinstance(c.args[0], str) and "INSERT INTO stock_prices" in c.args[0]
        ]
        self.assertEqual(insert_calls[0].args[1][0], "SYM_L")

    def test_handles_missing_optional_columns(self):
        """Should still work if ouverture / plus_bas etc. are missing."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        df = pd.DataFrame({
            "libelle": ["X"],
            "seance": [pd.Timestamp("2025-03-01")],
            "cloture": [42.0],
        })
        count = sink.persist_silver_prices(df)
        self.assertEqual(count, 1)

    def test_rollback_on_db_error(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.side_effect = [(1,), (1,)]
        # Make execute fail on INSERT
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First 2 calls: health-check, constraint-check. Then fail on INSERT
            if call_count[0] > 2:
                raise Exception("disk full")

        cursor.execute.side_effect = side_effect
        result = sink.persist_silver_prices(self._sample_df(3))
        conn.rollback.assert_called_once()
        self.assertEqual(result, 0)


# ══════════════════════════════════════════════════════════════════════
# _prepare_silver_rows
# ══════════════════════════════════════════════════════════════════════


class TestPrepareSilverRows(unittest.TestCase):

    def test_converts_to_tuples(self):
        df = pd.DataFrame({
            "libelle": ["BIAT", "SFBT"],
            "seance": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ouverture": [100.0, 200.0],
            "cloture": [105.0, 210.0],
            "plus_bas": [98.0, 195.0],
            "plus_haut": [108.0, 215.0],
            "quantite_negociee": [5000, 8000],
        })
        rows = DatabaseSink._prepare_silver_rows(df, "libelle")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "BIAT")
        self.assertEqual(rows[0][1], date(2025, 1, 2))
        self.assertIsInstance(rows[0][2], Decimal)  # ouverture
        self.assertIsInstance(rows[0][3], Decimal)  # cloture
        self.assertEqual(rows[1][6], 8000)           # quantite

    def test_handles_nan_values(self):
        df = pd.DataFrame({
            "code": ["BIAT"],
            "seance": pd.to_datetime(["2025-01-02"]),
            "ouverture": [float("nan")],
            "cloture": [105.0],
            "plus_bas": [float("nan")],
            "plus_haut": [float("nan")],
            "quantite_negociee": [float("nan")],
        })
        rows = DatabaseSink._prepare_silver_rows(df, "code")
        self.assertIsNone(rows[0][2])  # ouverture → None
        self.assertIsNone(rows[0][4])  # plus_bas → None
        self.assertIsNone(rows[0][5])  # plus_haut → None
        self.assertEqual(rows[0][6], 0)  # volume NaN → 0


# ══════════════════════════════════════════════════════════════════════
# Constraint helpers
# ══════════════════════════════════════════════════════════════════════


class TestConstraintHelpers(unittest.TestCase):

    def test_adds_price_prediction_constraint_when_missing(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        # These methods receive conn directly (no health-check).
        # fetchone is called once for the constraint check → None means missing
        cursor.fetchone.return_value = None
        sink._ensure_price_predictions_upsert_constraint(conn)

        alter_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and "ALTER TABLE" in str(c.args[0])
        ]
        self.assertEqual(len(alter_calls), 1)
        self.assertIn("price_predictions_symbol_target_model_uq", alter_calls[0].args[0])

    def test_skips_price_prediction_constraint_when_exists(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.return_value = (1,)  # constraint exists
        sink._ensure_price_predictions_upsert_constraint(conn)

        alter_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and "ALTER TABLE" in str(c.args[0])
        ]
        self.assertEqual(len(alter_calls), 0)

    def test_adds_watermark_constraint_when_missing(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.return_value = None
        sink._ensure_watermark_upsert_constraint(conn)

        alter_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and "ALTER TABLE" in str(c.args[0])
        ]
        self.assertEqual(len(alter_calls), 1)
        self.assertIn("etl_watermarks_layer_ticker_uq", alter_calls[0].args[0])

    def test_adds_stock_prices_constraint_when_missing(self):
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.return_value = None
        sink._ensure_stock_prices_upsert_constraint(conn)

        alter_calls = [
            c for c in cursor.execute.call_args_list
            if c.args and "ALTER TABLE" in str(c.args[0])
        ]
        self.assertEqual(len(alter_calls), 1)
        self.assertIn("stock_prices_symbol_seance_uq", alter_calls[0].args[0])

    def test_constraint_errors_are_silenced(self):
        """If ALTER TABLE fails (e.g. already exists under different name), no crash."""
        sink, conn, cursor = _make_sink_with_mock_conn()
        cursor.fetchone.return_value = None
        cursor.execute.side_effect = [None, Exception("already exists")]
        # Should not raise
        sink._ensure_price_predictions_upsert_constraint(conn)


# ══════════════════════════════════════════════════════════════════════
# Graceful Degradation
# ══════════════════════════════════════════════════════════════════════


class TestGracefulDegradation(unittest.TestCase):
    """All persist methods must return 0/False when DB is unavailable."""

    def test_all_methods_safe_without_db(self):
        sink = DatabaseSink()
        with patch.object(sink, "_get_connection", return_value=None):
            self.assertEqual(sink.persist_price_predictions([{"symbol": "X"}]), 0)
            self.assertEqual(sink.persist_volume_predictions([]), 0)
            self.assertEqual(sink.persist_liquidity_predictions([]), 0)
            self.assertFalse(sink.persist_model_metrics("M", "T", {}))
            self.assertFalse(sink.persist_watermark("silver", "T", date.today(), 0))
            self.assertEqual(sink.persist_silver_prices(pd.DataFrame()), 0)
            self.assertFalse(sink.ensure_tables())


# ══════════════════════════════════════════════════════════════════════
# Constructor
# ══════════════════════════════════════════════════════════════════════


class TestConstructor(unittest.TestCase):

    def test_default_params(self):
        sink = DatabaseSink()
        self.assertEqual(sink._conn_params["host"], "localhost")
        self.assertEqual(sink._conn_params["port"], 5432)
        self.assertEqual(sink._conn_params["database"], "fixtrade")
        self.assertEqual(sink._conn_params["user"], "postgres")
        self.assertEqual(sink._conn_params["password"], "")
        self.assertIsNone(sink._conn)

    def test_custom_params(self):
        sink = DatabaseSink(host="db.local", port=5433, database="test", user="app", password="secret")
        self.assertEqual(sink._conn_params["host"], "db.local")
        self.assertEqual(sink._conn_params["port"], 5433)
        self.assertEqual(sink._conn_params["database"], "test")
        self.assertEqual(sink._conn_params["user"], "app")
        self.assertEqual(sink._conn_params["password"], "secret")


if __name__ == "__main__":
    unittest.main()
