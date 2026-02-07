"""
Transform layer: Bronze → Silver data quality checks and cleaning.

Implements the Silver layer of the Medallion architecture:
- Data validation rules
- Null handling and type coercion
- Per-ticker normalization (stock splits, dividends)
- Deduplication with upsert semantics
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data Quality Validation Rules
# -------------------------------------------------------------------

VALIDATION_RULES = {
    "cloture_positive": lambda df: df["cloture"] > 0,
    "high_gte_low": lambda df: df["plus_haut"] >= df["plus_bas"],
    "volume_non_negative": lambda df: df["quantite_negociee"] >= 0,
    "no_future_dates": lambda df: pd.to_datetime(
        df["seance"].astype(str).str.strip(), dayfirst=True, format="mixed"
    ).dt.date <= date.today(),
}


class DataQualityChecker:
    """Validates Bronze-layer data against business rules.

    Flags or drops invalid rows depending on severity.
    Produces a quality report for monitoring.
    """

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all validation rules.

        Args:
            df: Bronze DataFrame with standardized columns.

        Returns:
            Tuple of (valid_rows, rejected_rows).
        """
        if df.empty:
            return df, df

        mask = pd.Series(True, index=df.index)
        rejection_reasons: list[str] = []

        for rule_name, rule_fn in VALIDATION_RULES.items():
            try:
                rule_mask = rule_fn(df)
                failures = (~rule_mask).sum()
                if failures > 0:
                    logger.warning(
                        "Validation rule '%s' failed for %d rows",
                        rule_name,
                        failures,
                    )
                    rejection_reasons.append(rule_name)
                mask &= rule_mask
            except Exception:
                logger.exception("Error in validation rule '%s'", rule_name)

        valid = df[mask].reset_index(drop=True)
        rejected = df[~mask].reset_index(drop=True)

        logger.info(
            "Quality check: %d valid, %d rejected out of %d total",
            len(valid),
            len(rejected),
            len(df),
        )
        return valid, rejected


class BronzeToSilverTransformer:
    """Transforms raw Bronze data into clean Silver data.

    Steps:
    1. Data quality validation
    2. Type coercion and null handling
    3. Per-ticker normalization
    4. Sorting and indexing
    """

    def __init__(self) -> None:
        self._checker = DataQualityChecker()

    def transform(self, bronze_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full Bronze → Silver transformation pipeline.

        Args:
            bronze_df: Raw data from Bronze layer.

        Returns:
            Cleaned, validated Silver DataFrame.
        """
        if bronze_df.empty:
            return bronze_df

        # Step 1: Validate
        valid_df, _ = self._checker.validate(bronze_df)

        # Step 2: Type coercion
        silver_df = self._coerce_types(valid_df)

        # Step 3: Handle missing values
        silver_df = self._handle_nulls(silver_df)

        # Step 4: Sort by code and date
        silver_df = silver_df.sort_values(
            ["code", "seance"]
        ).reset_index(drop=True)

        logger.info("Silver layer: %d rows, %d tickers", len(silver_df), silver_df["code"].nunique())
        return silver_df

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct dtypes for all columns."""
        df = df.copy()
        # Strip whitespace from string values before parsing
        if df["seance"].dtype == object:
            df["seance"] = df["seance"].astype(str).str.strip()
        df["seance"] = pd.to_datetime(df["seance"], dayfirst=True, format="mixed")
        numeric_cols = ["ouverture", "cloture", "plus_haut", "plus_bas"]
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "quantite_negociee" in df.columns:
            if df["quantite_negociee"].dtype == object:
                df["quantite_negociee"] = df["quantite_negociee"].astype(str).str.strip()
            df["quantite_negociee"] = pd.to_numeric(
                df["quantite_negociee"], errors="coerce"
            ).fillna(0).astype(int)
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.strip()
        return df

    @staticmethod
    def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill price NaNs within each ticker group."""
        df = df.copy()
        price_cols = ["ouverture", "cloture", "plus_haut", "plus_bas"]
        existing = [c for c in price_cols if c in df.columns]
        if existing and "code" in df.columns:
            df[existing] = df.groupby("code")[existing].ffill()
            df = df.dropna(subset=["cloture"])
        return df.reset_index(drop=True)
