"""
Extract layer: ingest raw BVMT data from CSV files and external sources.

Supports:
- Historical CSV files (BVMT Archive)
- Daily updates (API/scraping results)
- External data (TUNINDEX, Forex, News)

All extracted data is stored as-is in the Bronze layer (immutable, schema-on-read).
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from prediction.config import PredictionConfig, config

logger = logging.getLogger(__name__)

# Expected column mapping: BVMT raw CSV → standardized names
BVMT_COLUMN_MAP = {
    "SEANCE": "seance",
    "CODE": "code",
    "LIBELLE": "libelle",
    "VALEUR": "libelle",
    "OUVERTURE": "ouverture",
    "CLOTURE": "cloture",
    "PLUS_HAUT": "plus_haut",
    "PLUS_BAS": "plus_bas",
    "QUANTITE_NEGOCIEE": "quantite_negociee",
    "VOLUME": "volume",
    "VARIATION": "variation",
    "CAPITAUX": "capitaux",
    "NB_TRANSACTION": "nb_transaction",
    "GROUPE": "groupe",
    # Alternative column names sometimes seen in exports
    "Date": "seance",
    "Open": "ouverture",
    "Close": "cloture",
    "High": "plus_haut",
    "Low": "plus_bas",
    "Volume": "quantite_negociee",
}


class BVMTExtractor:
    """Extracts raw BVMT trade data from CSV files into Bronze layer.

    Implements idempotent extraction: re-running on the same files
    will overwrite the Bronze partitions without creating duplicates.
    """

    def __init__(self, cfg: PredictionConfig | None = None) -> None:
        self._cfg = cfg or config
        self._bronze_path = self._cfg.paths.bronze
        self._raw_path = self._cfg.paths.raw

    def extract_all_csv(self) -> pd.DataFrame:
        """Read all CSV files from data/raw/ and combine into a single DataFrame.

        Returns:
            Combined DataFrame with standardized column names.
        """
        csv_files = sorted(self._raw_path.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", self._raw_path)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for csv_file in csv_files:
            logger.info("Extracting: %s", csv_file.name)
            try:
                df = self._read_single_csv(csv_file)
                frames.append(df)
            except Exception:
                logger.exception("Failed to read %s", csv_file.name)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._standardize_columns(combined)
        combined = self._deduplicate(combined)

        logger.info(
            "Extracted %d rows from %d files", len(combined), len(frames)
        )
        return combined

    def extract_incremental(self, since: date) -> pd.DataFrame:
        """Extract only data newer than the given watermark date.

        Args:
            since: Only return rows where seance > since.

        Returns:
            DataFrame of new records.
        """
        df = self.extract_all_csv()
        if df.empty:
            return df

        df["seance"] = pd.to_datetime(df["seance"]).dt.date
        return df[df["seance"] > since].reset_index(drop=True)

    def save_to_bronze(self, df: pd.DataFrame) -> None:
        """Persist extracted data to Bronze layer as partitioned Parquet.

        Partition scheme: /year=YYYY/month=MM/code=XXXXXX/data.parquet
        """
        if df.empty:
            logger.warning("Empty DataFrame — nothing to save to Bronze.")
            return

        df = df.copy()
        df["seance"] = pd.to_datetime(df["seance"])
        df["year"] = df["seance"].dt.year
        df["month"] = df["seance"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "code"]):
            partition_dir = (
                self._bronze_path / f"year={year}" / f"month={month:02d}" / f"code={code}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            out_path = partition_dir / "data.parquet"
            group.drop(columns=["year", "month"]).to_parquet(
                out_path, index=False, engine="pyarrow"
            )

        logger.info("Bronze layer written to %s", self._bronze_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_single_csv(path: Path) -> pd.DataFrame:
        """Read a single CSV with flexible encoding and delimiter detection."""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            for sep in (",", ";", "\t"):
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        low_memory=False,
                    )
                    if len(df.columns) > 2:
                        # Strip whitespace from column names (BVMT exports have trailing spaces)
                        df.columns = df.columns.str.strip()
                        return df
                except Exception:
                    continue
        raise ValueError(f"Could not parse CSV: {path}")

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match the standardized schema."""
        rename_map = {
            col: BVMT_COLUMN_MAP[col]
            for col in df.columns
            if col in BVMT_COLUMN_MAP
        }
        return df.rename(columns=rename_map)

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicates, keeping the first occurrence."""
        before = len(df)
        key_cols = [c for c in ("seance", "code") if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")
        else:
            df = df.drop_duplicates(keep="first")
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows", removed)
        return df.reset_index(drop=True)
