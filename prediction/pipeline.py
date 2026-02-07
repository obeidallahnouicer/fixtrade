"""
ETL pipeline orchestrator.

Runs the full Extract → Transform → Load pipeline:
1. Extract raw CSV data from data/raw/
2. Bronze layer: store raw data (immutable, partitioned)
3. Silver layer: validate, clean, enrich
4. Feature engineering: 50+ technical/temporal/volume features
5. Gold layer: create ML-ready train/val/test views

Supports incremental processing via watermark tracking.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from prediction.config import PredictionConfig, config
from prediction.etl.extract.bvmt_extractor import BVMTExtractor
from prediction.etl.load.parquet_loader import ParquetLoader
from prediction.etl.transform.bronze_to_silver import BronzeToSilverTransformer
from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
from prediction.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class ETLPipeline:
    """End-to-end ETL pipeline orchestrator.

    Usage:
        pipeline = ETLPipeline()
        gold_df = pipeline.run()  # Full pipeline
        # or
        gold_df = pipeline.run_incremental()  # Only new data
    """

    def __init__(self, cfg: PredictionConfig | None = None) -> None:
        self._cfg = cfg or config
        self._extractor = BVMTExtractor(cfg=self._cfg)
        self._loader = ParquetLoader(self._cfg.paths.base_dir)
        self._bronze_transformer = BronzeToSilverTransformer()
        self._feature_pipeline = FeaturePipeline(cfg=self._cfg.features)
        self._gold_transformer = SilverToGoldTransformer(
            train_end_year=self._cfg.model.train_test_split_year - 1,
            val_end_year=self._cfg.model.train_test_split_year,
            prediction_horizons=self._cfg.features.prediction_horizons,
        )

    def run(self) -> pd.DataFrame:
        """Execute the full ETL pipeline.

        Returns:
            Gold-layer DataFrame with features and targets.
        """
        logger.info("=" * 60)
        logger.info("Starting full ETL pipeline")
        logger.info("=" * 60)

        # 1. Extract
        logger.info("Phase 1: EXTRACT")
        raw_df = self._extractor.extract_all_csv()
        if raw_df.empty:
            logger.warning("No data extracted. Pipeline aborted.")
            return pd.DataFrame()

        # 2. Bronze
        logger.info("Phase 2: BRONZE (raw storage)")
        self._loader.save_partitioned(
            raw_df, layer="bronze", partition_cols=["code"]
        )

        # 3. Silver
        logger.info("Phase 3: SILVER (clean + validate)")
        silver_df = self._bronze_transformer.transform(raw_df)
        if silver_df.empty:
            logger.warning("Silver layer is empty after transformation.")
            return pd.DataFrame()

        # 4. Feature Engineering
        logger.info("Phase 4: FEATURE ENGINEERING")
        enriched_df = self._feature_pipeline.run(silver_df)

        # 5. Gold
        logger.info("Phase 5: GOLD (ML-ready views)")
        self._loader.save_partitioned(
            enriched_df, layer="silver", partition_cols=["code"]
        )

        # Update watermark
        if "seance" in enriched_df.columns:
            max_date = pd.to_datetime(enriched_df["seance"]).max().date()
            self._loader.set_watermark("silver", max_date)

        logger.info("ETL pipeline complete. %d rows ready.", len(enriched_df))
        return enriched_df

    def run_incremental(self) -> pd.DataFrame:
        """Run incremental ETL — only process new data.

        Uses watermark-based tracking to avoid reprocessing.

        Returns:
            Newly processed Gold-layer DataFrame.
        """
        watermark = self._loader.get_watermark("silver")
        if watermark is None:
            logger.info("No watermark found. Running full pipeline.")
            return self.run()

        logger.info("Incremental ETL since %s", watermark)

        # Extract only new data
        new_data = self._extractor.extract_incremental(since=watermark)
        if new_data.empty:
            logger.info("No new data since %s. Skipping.", watermark)
            return pd.DataFrame()

        # Process through pipeline
        silver_new = self._bronze_transformer.transform(new_data)
        if silver_new.empty:
            return pd.DataFrame()

        # For feature engineering, we need historical context (lookback window)
        existing_silver = self._loader.load_layer("silver")
        if not existing_silver.empty:
            combined = pd.concat([existing_silver, silver_new], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["code", "seance"], keep="last"
            ).sort_values(["code", "seance"]).reset_index(drop=True)
        else:
            combined = silver_new

        enriched = self._feature_pipeline.run(combined)

        # Save updated silver layer
        self._loader.save_partitioned(
            enriched, layer="silver", partition_cols=["code"]
        )

        # Update watermark
        if "seance" in silver_new.columns:
            max_date = pd.to_datetime(silver_new["seance"]).max().date()
            self._loader.set_watermark("silver", max_date)

        logger.info(
            "Incremental ETL: %d new rows processed.", len(silver_new)
        )
        return enriched

    def get_gold_datasets(
        self, features_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train/val/test splits from the Gold layer.

        Args:
            features_df: Pre-computed features. If None, loads from Silver.

        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        if features_df is None:
            features_df = self._loader.load_layer("silver")

        if features_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        return self._gold_transformer.create_training_view(features_df)
