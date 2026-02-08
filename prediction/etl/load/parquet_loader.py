"""
Load layer: persist data to Bronze / Silver / Gold storage.

Supports:
- Parquet files (columnar, ~80% compression)
- Partitioned storage (year/month/code)
- Watermark-based incremental updates
- Metadata tracking for data lineage
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetLoader:
    """Loads DataFrames into partitioned Parquet storage.

    Supports the three Medallion layers with consistent
    partitioning and metadata tracking.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._metadata_path = base_path / "_metadata.json"
        self._metadata = self._load_metadata()

    def save_partitioned(
        self,
        df: pd.DataFrame,
        layer: str,
        partition_cols: list[str] | None = None,
    ) -> None:
        """Save a DataFrame to the specified layer as Parquet.

        Args:
            df: Data to persist.
            layer: One of 'bronze', 'silver', 'gold'.
            partition_cols: Columns to partition by.
        """
        if df.empty:
            logger.warning("Empty DataFrame — skipping save to %s.", layer)
            return

        layer_path = self._base_path / layer
        layer_path.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            for col in partition_cols:
                if col not in df.columns:
                    logger.warning("Partition column '%s' not in DataFrame.", col)
                    partition_cols.remove(col)

        if partition_cols:
            self._save_partitioned_by_columns(df, layer_path, partition_cols)
        else:
            out_path = layer_path / "data.parquet"
            df.to_parquet(out_path, index=False, engine="pyarrow")

        # Update metadata
        self._update_metadata(layer, len(df))
        logger.info("Saved %d rows to %s layer.", len(df), layer)

    def load_layer(self, layer: str) -> pd.DataFrame:
        """Load all data from a Medallion layer.

        Reconstructs partition columns (e.g. ``code``) from the
        Hive-style directory structure ``col=value/``.

        Args:
            layer: One of 'bronze', 'silver', 'gold'.

        Returns:
            Combined DataFrame from all Parquet files in the layer.
        """
        layer_path = self._base_path / layer
        if not layer_path.exists():
            logger.warning("Layer path does not exist: %s", layer_path)
            return pd.DataFrame()

        parquet_files = sorted(layer_path.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            # Reconstruct partition columns from Hive-style path parts
            # e.g. .../silver/code=BIAT/data.parquet → code="BIAT"
            rel = pf.relative_to(layer_path)
            for part in rel.parts[:-1]:  # skip the filename
                if "=" in part:
                    col, val = part.split("=", 1)
                    if col not in df.columns:
                        df[col] = val
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def get_watermark(self, layer: str) -> date | None:
        """Get the last processed date for a layer.

        Returns:
            The watermark date, or None if not yet tracked.
        """
        key = f"{layer}_last_processed_date"
        val = self._metadata.get(key)
        if val:
            return date.fromisoformat(val)
        return None

    def set_watermark(self, layer: str, watermark: date) -> None:
        """Update the watermark for a layer."""
        key = f"{layer}_last_processed_date"
        self._metadata[key] = watermark.isoformat()
        self._save_metadata()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_partitioned_by_columns(
        df: pd.DataFrame, layer_path: Path, partition_cols: list[str]
    ) -> None:
        """Write Parquet files partitioned by the given columns."""
        for keys, group in df.groupby(partition_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)

            parts = [f"{col}={val}" for col, val in zip(partition_cols, keys)]
            part_dir = layer_path / Path(*parts)
            part_dir.mkdir(parents=True, exist_ok=True)

            out = group.drop(columns=partition_cols, errors="ignore")

            # Fix mixed-type object columns → str (avoids PyArrow errors)
            for col in out.columns:
                if out[col].dtype == "object":
                    out[col] = out[col].astype(str)

            out.to_parquet(
                part_dir / "data.parquet",
                index=False,
                engine="pyarrow",
            )

    def _load_metadata(self) -> dict:
        """Load metadata from disk."""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Persist metadata to disk."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _update_metadata(self, layer: str, row_count: int) -> None:
        """Track layer update stats in metadata."""
        self._metadata[f"{layer}_last_updated"] = datetime.now().isoformat()
        self._metadata[f"{layer}_row_count"] = row_count
        self._save_metadata()
