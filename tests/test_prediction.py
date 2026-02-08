"""
Tests for the prediction module.

Covers:
- Configuration loading
- Feature engineering (technical, temporal, volume, lag)
- Data quality validation (Bronze → Silver)
- Gold layer dataset creation
- Model base class interface
- Ensemble weighting logic
- Cache client (in-memory fallback)
- ETL pipeline integration
- Anti-leakage verification
"""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame mimicking BVMT data."""
    np.random.seed(42)
    n_days = 300
    dates = pd.bdate_range(start="2023-01-02", periods=n_days, freq="B")
    base_price = 50.0
    prices = base_price + np.cumsum(np.random.randn(n_days) * 0.5)
    prices = np.maximum(prices, 1.0)  # No negative prices

    return pd.DataFrame({
        "seance": dates,
        "code": "BIAT",
        "libelle": "BIAT",
        "ouverture": prices + np.random.rand(n_days) * 0.5,
        "cloture": prices,
        "plus_haut": prices + np.abs(np.random.randn(n_days) * 0.3),
        "plus_bas": prices - np.abs(np.random.randn(n_days) * 0.3),
        "quantite_negociee": np.random.randint(1000, 50000, n_days),
    })


@pytest.fixture
def multi_ticker_ohlcv() -> pd.DataFrame:
    """OHLCV data with multiple tickers spanning 2020-2025."""
    np.random.seed(123)
    frames = []
    for code in ["BIAT", "SFBT", "DELICE"]:
        n = 1200
        dates = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
        base = np.random.uniform(20, 100)
        prices = base + np.cumsum(np.random.randn(n) * 0.5)
        prices = np.maximum(prices, 1.0)
        df = pd.DataFrame({
            "seance": dates,
            "code": code,
            "libelle": code,
            "ouverture": prices + 0.3,
            "cloture": prices,
            "plus_haut": prices + np.abs(np.random.randn(n) * 0.5),
            "plus_bas": prices - np.abs(np.random.randn(n) * 0.5),
            "quantite_negociee": np.random.randint(500, 30000, n),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config_loads(self):
        from prediction.config import PredictionConfig
        cfg = PredictionConfig()
        assert cfg.paths.bronze.name == "bronze"
        assert cfg.model.lstm_sequence_length == 60
        assert cfg.features.rsi_window == 14
        assert len(cfg.tracked_tickers) >= 20

    def test_paths_are_consistent(self):
        from prediction.config import PredictionConfig
        cfg = PredictionConfig()
        assert cfg.paths.bronze.parent == cfg.paths.silver.parent
        assert cfg.paths.gold.parent == cfg.paths.bronze.parent

    def test_ensemble_weights_sum_to_one(self):
        from prediction.config import PredictionConfig
        cfg = PredictionConfig()
        total = (
            cfg.model.ensemble_lstm_weight
            + cfg.model.ensemble_xgb_weight
            + cfg.model.ensemble_prophet_weight
        )
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Feature Engineering tests
# ---------------------------------------------------------------------------


class TestTechnicalFeatures:
    def test_computes_sma(self, sample_ohlcv):
        from prediction.features.technical import TechnicalFeatures
        tf = TechnicalFeatures(sma_windows=(5, 20))
        result = tf.compute(sample_ohlcv)
        assert "sma_5" in result.columns
        assert "sma_20" in result.columns
        # First 20 rows should be NaN (window + shift)
        assert result["sma_20"].isna().sum() >= 20

    def test_computes_rsi(self, sample_ohlcv):
        from prediction.features.technical import TechnicalFeatures
        tf = TechnicalFeatures()
        result = tf.compute(sample_ohlcv)
        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        valid_rsi = result["rsi"].dropna()
        assert valid_rsi.between(0, 100).all()

    def test_computes_macd(self, sample_ohlcv):
        from prediction.features.technical import TechnicalFeatures
        tf = TechnicalFeatures()
        result = tf.compute(sample_ohlcv)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_all_features_shifted(self, sample_ohlcv):
        """Anti-leakage: all technical features should use .shift(1)."""
        from prediction.features.technical import TechnicalFeatures
        tf = TechnicalFeatures(sma_windows=(5,))
        result = tf.compute(sample_ohlcv)
        # The first row of any shifted feature should be NaN
        assert pd.isna(result["sma_5"].iloc[0])

    def test_no_future_data_leakage(self, sample_ohlcv):
        """Verify features at time t don't use data from time t."""
        from prediction.features.technical import TechnicalFeatures
        tf = TechnicalFeatures(sma_windows=(5,))
        result = tf.compute(sample_ohlcv)

        # SMA_5 at index i should be computed from indices [i-6, i-2] (shifted by 1)
        # So SMA_5 at row 6 = mean of close[0:5], not close[1:6]
        manual_sma = sample_ohlcv["cloture"].iloc[:5].mean()
        assert abs(result["sma_5"].iloc[5] - manual_sma) < 1e-6


class TestTemporalFeatures:
    def test_computes_calendar_features(self, sample_ohlcv):
        from prediction.features.temporal import TemporalFeatures
        tf = TemporalFeatures()
        result = tf.compute(sample_ohlcv)
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_month_end" in result.columns
        assert "day_of_week_sin" in result.columns

    def test_cyclical_encoding_bounded(self, sample_ohlcv):
        from prediction.features.temporal import TemporalFeatures
        tf = TemporalFeatures()
        result = tf.compute(sample_ohlcv)
        assert result["day_of_week_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()


class TestVolumeFeatures:
    def test_computes_volume_features(self, sample_ohlcv):
        from prediction.features.volume import VolumeFeatures
        vf = VolumeFeatures()
        result = vf.compute(sample_ohlcv)
        assert "volume_sma_5" in result.columns
        assert "volume_ratio" in result.columns
        assert "vwap" in result.columns
        assert "mfi" in result.columns

    def test_features_shifted(self, sample_ohlcv):
        from prediction.features.volume import VolumeFeatures
        vf = VolumeFeatures()
        result = vf.compute(sample_ohlcv)
        assert pd.isna(result["volume_sma_5"].iloc[0])


class TestLagFeatures:
    def test_computes_lag_features(self, sample_ohlcv):
        from prediction.features.lag import LagFeatures
        lf = LagFeatures(lag_days=(1, 5))
        result = lf.compute(sample_ohlcv)
        assert "close_lag_1" in result.columns
        assert "close_lag_5" in result.columns
        assert "return_lag_1" in result.columns
        assert "momentum_5d" in result.columns
        assert "mean_reversion_z" in result.columns


class TestFeaturePipeline:
    def test_full_pipeline(self, sample_ohlcv):
        from prediction.features.pipeline import FeaturePipeline
        fp = FeaturePipeline()
        result = fp.run(sample_ohlcv)
        # Should have many features
        assert len(result.columns) > 30
        assert len(result) == len(sample_ohlcv)


# ---------------------------------------------------------------------------
# Data Quality / Transform tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    def test_validates_positive_close(self):
        from prediction.etl.transform.bronze_to_silver import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame({
            "cloture": [10, -5, 20],
            "plus_haut": [12, 6, 22],
            "plus_bas": [8, 4, 18],
            "quantite_negociee": [100, 200, 300],
            "seance": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        valid, rejected = checker.validate(df)
        assert len(valid) == 2  # Row with cloture=-5 rejected
        assert len(rejected) == 1

    def test_validates_high_gte_low(self):
        from prediction.etl.transform.bronze_to_silver import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame({
            "cloture": [10, 20],
            "plus_haut": [12, 18],  # Second row: high < low
            "plus_bas": [8, 22],
            "quantite_negociee": [100, 200],
            "seance": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        })
        valid, rejected = checker.validate(df)
        assert len(valid) == 1
        assert len(rejected) == 1


class TestBronzeToSilver:
    def test_transforms_bronze_data(self, sample_ohlcv):
        from prediction.etl.transform.bronze_to_silver import BronzeToSilverTransformer
        transformer = BronzeToSilverTransformer()
        silver = transformer.transform(sample_ohlcv)
        assert not silver.empty
        assert silver["cloture"].dtype in (np.float64, np.float32)

    def test_sorted_by_code_and_date(self, multi_ticker_ohlcv):
        from prediction.etl.transform.bronze_to_silver import BronzeToSilverTransformer
        transformer = BronzeToSilverTransformer()
        silver = transformer.transform(multi_ticker_ohlcv)
        # Check sorting within first ticker
        biat = silver[silver["code"] == "BIAT"]
        assert (biat["seance"].diff().dropna() >= pd.Timedelta(0)).all()


# ---------------------------------------------------------------------------
# Gold Layer tests
# ---------------------------------------------------------------------------


class TestSilverToGold:
    def test_creates_training_view(self, multi_ticker_ohlcv):
        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        gt = SilverToGoldTransformer(
            train_end_year=2023,
            val_end_year=2024,
        )
        multi_ticker_ohlcv["seance"] = pd.to_datetime(multi_ticker_ohlcv["seance"])
        train, val, test = gt.create_training_view(multi_ticker_ohlcv)
        assert not train.empty
        assert not val.empty
        # Train years <= 2023
        assert train["seance"].dt.year.max() <= 2023
        # Val years in (2023, 2024]
        assert val["seance"].dt.year.min() >= 2024

    def test_adds_targets(self, sample_ohlcv):
        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        gt = SilverToGoldTransformer()
        sample_ohlcv["seance"] = pd.to_datetime(sample_ohlcv["seance"])
        result = gt._add_targets(sample_ohlcv)
        assert "target_1d" in result.columns
        assert "target_5d" in result.columns
        # Last row's target should be NaN
        assert pd.isna(result["target_1d"].iloc[-1])

    def test_inference_view_latest_row(self, multi_ticker_ohlcv):
        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        gt = SilverToGoldTransformer()
        multi_ticker_ohlcv["seance"] = pd.to_datetime(multi_ticker_ohlcv["seance"])
        inference = gt.create_inference_view(multi_ticker_ohlcv)
        # One row per ticker
        assert len(inference) == multi_ticker_ohlcv["code"].nunique()


# ---------------------------------------------------------------------------
# Model Base Class tests
# ---------------------------------------------------------------------------


class TestModelMetrics:
    def test_compute_metrics(self):
        from prediction.models.base import BasePredictionModel
        y_true = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        y_pred = np.array([10.1, 11.2, 11.8, 13.1, 14.3])
        metrics = BasePredictionModel._compute_metrics(y_true, y_pred)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert 0 <= metrics.mape <= 1
        assert 0 <= metrics.directional_accuracy <= 1
        assert metrics.r_squared > 0.9  # Should be high for close predictions


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_liquidity_tier_classification(self):
        from prediction.models.ensemble import LiquidityTier
        assert LiquidityTier.classify(15000) == LiquidityTier.TIER1
        assert LiquidityTier.classify(5000) == LiquidityTier.TIER2
        assert LiquidityTier.classify(500) == LiquidityTier.TIER3

    def test_model_selection_by_tier(self):
        from prediction.models.ensemble import EnsemblePredictor
        models = EnsemblePredictor._select_models_for_tier("high")
        assert "LSTM" in models
        assert "XGBoost" in models
        assert "Prophet" in models

        models = EnsemblePredictor._select_models_for_tier("medium")
        assert "LSTM" not in models
        assert "XGBoost" in models

        models = EnsemblePredictor._select_models_for_tier("low")
        assert models == ["Prophet"]

    def test_dynamic_weights_high_volatility(self):
        from prediction.models.ensemble import EnsemblePredictor
        weights = EnsemblePredictor._compute_dynamic_weights(0.05, 10000)
        assert weights["Prophet"] > weights["LSTM"]  # Prophet should dominate

    def test_weight_normalization(self):
        from prediction.models.ensemble import EnsemblePredictor
        ens = EnsemblePredictor(weights={"A": 2.0, "B": 3.0})
        ens._normalize_weights()
        total = sum(ens._weights.values())
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCache:
    def test_in_memory_cache(self):
        from prediction.utils.cache import CacheClient
        # Use invalid URL to force in-memory fallback
        cache = CacheClient(redis_url="redis://invalid:9999/0")
        cache.set_prediction("BIAT", {"price": 50.0})
        result = cache.get_prediction("BIAT")
        assert result is not None
        assert result["price"] == 50.0

    def test_cache_invalidation(self):
        from prediction.utils.cache import CacheClient
        cache = CacheClient(redis_url="redis://invalid:9999/0")
        cache.set_prediction("BIAT", {"price": 50.0})
        cache.set_prediction("SFBT", {"price": 30.0})
        deleted = cache.invalidate_predictions("BIAT")
        assert deleted >= 1
        assert cache.get_prediction("BIAT") is None
        assert cache.get_prediction("SFBT") is not None

    def test_feature_store(self):
        from prediction.utils.cache import CacheClient
        cache = CacheClient(redis_url="redis://invalid:9999/0")
        features = {"rsi": 65.2, "macd": 0.5, "sma_20": 48.3}
        cache.set_features("BIAT", "2024-01-15", features)
        result = cache.get_features("BIAT", "2024-01-15")
        assert result is not None
        assert result["rsi"] == 65.2


# ---------------------------------------------------------------------------
# Metrics / Monitor tests
# ---------------------------------------------------------------------------


class TestModelMonitor:
    def test_evaluate_model(self):
        from prediction.utils.metrics import ModelMonitor
        monitor = ModelMonitor(rmse_threshold=1.0)
        y_true = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        y_pred = np.array([10.1, 11.2, 11.8, 13.1, 14.3])
        report = monitor.evaluate_model("test_model", y_true, y_pred)
        assert report.model_name == "test_model"
        assert report.rmse > 0
        assert report.sample_size == 5

    def test_drift_detection(self):
        from prediction.utils.metrics import ModelMonitor
        monitor = ModelMonitor()
        # Large drift
        y_true = np.array([100.0, 101.0, 102.0])
        y_pred = np.array([80.0, 81.0, 82.0])
        report = monitor.evaluate_model("drift_model", y_true, y_pred)
        assert report.drift_detected is True

    def test_should_retrain_high_rmse(self):
        from prediction.utils.metrics import ModelMonitor
        monitor = ModelMonitor(rmse_threshold=0.1)
        y_true = np.array([10.0, 11.0, 12.0])
        y_pred = np.array([15.0, 16.0, 17.0])  # Very bad predictions
        monitor.evaluate_model("bad_model", y_true, y_pred)
        assert monitor.should_retrain("bad_model") is True


# ---------------------------------------------------------------------------
# Parquet Loader tests
# ---------------------------------------------------------------------------


class TestParquetLoader:
    def test_save_and_load(self, tmp_path, sample_ohlcv):
        from prediction.etl.load.parquet_loader import ParquetLoader
        loader = ParquetLoader(tmp_path)
        loader.save_partitioned(sample_ohlcv, "silver")
        loaded = loader.load_layer("silver")
        assert not loaded.empty
        assert len(loaded) == len(sample_ohlcv)

    def test_watermark_tracking(self, tmp_path):
        from prediction.etl.load.parquet_loader import ParquetLoader
        loader = ParquetLoader(tmp_path)
        assert loader.get_watermark("silver") is None
        loader.set_watermark("silver", date(2024, 6, 15))
        assert loader.get_watermark("silver") == date(2024, 6, 15)

    def test_partitioned_save(self, tmp_path, sample_ohlcv):
        from prediction.etl.load.parquet_loader import ParquetLoader
        loader = ParquetLoader(tmp_path)
        loader.save_partitioned(
            sample_ohlcv, "bronze", partition_cols=["code"]
        )
        bronze_path = tmp_path / "bronze"
        assert bronze_path.exists()
        parquet_files = list(bronze_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1


# ---------------------------------------------------------------------------
# Inference Service tests
# ---------------------------------------------------------------------------


class TestPredictionService:
    def test_fallback_prediction(self):
        from prediction.inference import PredictionService
        results = PredictionService._fallback_prediction("BIAT", 3)
        assert len(results) == 3
        assert all(r.symbol == "BIAT" for r in results)
        assert all(r.model_name == "fallback" for r in results)

    def test_next_trading_day_skips_weekends(self):
        from prediction.inference import PredictionService
        # Friday → next trading day should be Monday
        friday = date(2024, 1, 5)  # A Friday
        next_day = PredictionService._next_trading_day(friday, 1)
        assert next_day.weekday() == 0  # Monday

    def test_serialization_roundtrip(self):
        from prediction.inference import PredictionResult, PredictionService
        results = [
            PredictionResult(
                symbol="BIAT",
                target_date=date(2024, 1, 8),
                predicted_close=50.123,
                confidence_lower=48.5,
                confidence_upper=51.8,
            )
        ]
        serialized = PredictionService._serialize_predictions(results)
        deserialized = PredictionService._deserialize_predictions(serialized)
        assert len(deserialized) == 1
        assert deserialized[0].symbol == "BIAT"
        assert float(deserialized[0].predicted_close) == pytest.approx(50.123, rel=1e-3)


# ---------------------------------------------------------------------------
# Anti-Leakage Integration test
# ---------------------------------------------------------------------------


class TestAntiLeakage:
    """Verify no future data leaks into features."""

    def test_features_only_use_past_data(self, sample_ohlcv):
        """The feature at time t should ONLY depend on data from t-1 and earlier."""
        from prediction.features.pipeline import FeaturePipeline
        fp = FeaturePipeline()
        result = fp.run(sample_ohlcv)

        # Pick a feature column and verify it's shifted
        feature_cols = [
            c for c in result.columns
            if c.startswith("sma_") or c.startswith("rsi")
        ]
        for col in feature_cols:
            # The first row should always be NaN (due to shift(1))
            assert pd.isna(result[col].iloc[0]), (
                f"Feature '{col}' is not properly shifted — row 0 should be NaN"
            )

    def test_targets_are_future_looking(self, sample_ohlcv):
        """Targets should be future closing prices (shift(-h))."""
        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        gt = SilverToGoldTransformer()
        sample_ohlcv["seance"] = pd.to_datetime(sample_ohlcv["seance"])
        result = gt._add_targets(sample_ohlcv)

        # target_1d at row i should equal cloture at row i+1
        for i in range(len(result) - 1):
            if not pd.isna(result["target_1d"].iloc[i]):
                assert result["target_1d"].iloc[i] == result["cloture"].iloc[i + 1]
