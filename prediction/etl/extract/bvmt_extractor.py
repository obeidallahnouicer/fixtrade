"""
Extract layer: ingest raw BVMT data from CSV and fixed-width TXT files.

Supports:
- Historical CSV files (BVMT Archive, semicolon-separated)
- Historical TXT files (BVMT Archive, fixed-width format, 2016–2021)
- Daily updates (API/scraping results)
- External data (TUNINDEX, Forex, News)

All extracted data is stored as-is in the Bronze layer (immutable, schema-on-read).
"""

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from prediction.config import PredictionConfig, config

logger = logging.getLogger(__name__)

# Expected column mapping: BVMT raw CSV / TXT → standardized names
BVMT_COLUMN_MAP = {
    # Standard CSV headers
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
    # TXT headers (2016-2017, 2020-2021)
    "VALEUR": "libelle",
    "CAPITAUX": "capitaux",
    "GROUPE": "groupe",
    "NB_TRANSACTION": "nb_transaction",
    "IND_RES": "_ind_res",
    # TXT headers (2018-2019) — different column names
    "CODE_VAL": "code",
    "LIB_VAL": "libelle",
    "C_GR_RLC": "_groupe",
    "COURS_REF": "_cours_ref",
    "COURS_VEILLE": "_cours_veille",
    "DERNIER_COURS": "_dernier_cours",
    "NB_TRAN": "_nb_transaction",
    "I": "_ind_res",
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
        """Read all CSV and TXT files from data/raw/ and combine into a single DataFrame.

        Returns:
            Combined DataFrame with standardized column names.
        """
        csv_files = sorted(self._raw_path.glob("*.csv"))
        txt_files = sorted(self._raw_path.glob("*.txt"))
        all_files = csv_files + txt_files

        if not all_files:
            logger.warning("No CSV or TXT files found in %s", self._raw_path)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for data_file in all_files:
            logger.info("Extracting: %s", data_file.name)
            try:
                if data_file.suffix.lower() == ".txt":
                    df = self._read_fixed_width_txt(data_file)
                else:
                    df = self._read_single_csv(data_file)
                # Standardize columns per file before concatenation
                # to avoid duplicate column names across different schemas
                df = self._standardize_columns(df)
                frames.append(df)
            except Exception:
                logger.exception("Failed to read %s", data_file.name)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
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
    def _read_fixed_width_txt(path: Path) -> pd.DataFrame:
        """Read a BVMT fixed-width TXT file.

        Handles three format variants:
        - 2016-2017: SEANCE, GROUPE, CODE (numeric), VALEUR, ...
        - 2018-2019: SEANCE, C_GR_RLC, CODE_VAL (numeric), LIB_VAL, ...,
                     extra cols (COURS_REF, COURS_VEILLE, DERNIER_COURS),
                     2-digit years, comma decimal separators
        - 2020-2021: SEANCE, GROUPE, CODE (ISIN-style), VALEUR, ...

        Uses the separator line (------) to detect column boundaries.
        """
        raw_text = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(path, encoding=encoding) as f:
                    raw_text = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        if raw_text is None:
            raise ValueError(f"Could not decode TXT file: {path}")

        if len(raw_text) < 3:
            raise ValueError(f"TXT file too short: {path}")

        header_line = raw_text[0].rstrip()
        sep_line = raw_text[1].rstrip()

        # Build column specs from the separator line dash groups
        colspecs = [
            (m.start(), m.end()) for m in re.finditer(r"-+", sep_line)
        ]
        col_names = [header_line[s:e].strip() for s, e in colspecs]

        # Parse data lines (skip header + separator, ignore blank/dash lines)
        data_lines = [
            line
            for line in raw_text[2:]
            if line.strip() and not line.strip().startswith("-")
        ]

        if not data_lines:
            logger.warning("No data rows found in %s", path.name)
            return pd.DataFrame(columns=col_names)

        rows = [
            [line[s:e].strip() if s < len(line) else "" for s, e in colspecs]
            for line in data_lines
        ]
        df = pd.DataFrame(rows, columns=col_names)

        # 2018-2019 format uses comma decimals and 2-digit years
        is_2018_format = "CODE_VAL" in df.columns
        if is_2018_format:
            df = BVMTExtractor._fix_2018_format(df)

        # Coerce numeric columns
        for col in ("OUVERTURE", "CLOTURE", "PLUS_HAUT", "PLUS_BAS"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ("QUANTITE_NEGOCIEE", "CAPITAUX"):
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "Parsed TXT %s: %d rows, %d columns",
            path.name, len(df), len(df.columns),
        )
        return df

    @staticmethod
    def _fix_2018_format(df: pd.DataFrame) -> pd.DataFrame:
        """Fix 2018-2019 TXT format: comma decimals → dots, 2-digit → 4-digit year."""
        df = df.copy()
        comma_cols = [
            "COURS_REF", "COURS_VEILLE", "OUVERTURE", "DERNIER_COURS",
            "CLOTURE", "PLUS_BAS", "PLUS_HAUT", "CAPITAUX",
        ]
        for col in comma_cols:
            if col in df.columns:
                df[col] = df[col].str.replace(",", ".", regex=False)

        if "SEANCE" in df.columns:
            df["SEANCE"] = df["SEANCE"].apply(BVMTExtractor._fix_short_year)
        return df

    @staticmethod
    def _fix_short_year(date_str: str) -> str:
        """Convert 2-digit year dates to 4-digit (e.g. 02/01/18 → 02/01/2018)."""
        if not isinstance(date_str, str):
            return date_str
        parts = date_str.strip().split("/")
        if len(parts) == 3 and len(parts[2]) == 2:
            year_2d = int(parts[2])
            year_4d = 2000 + year_2d if year_2d < 50 else 1900 + year_2d
            return f"{parts[0]}/{parts[1]}/{year_4d}"
        return date_str

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
        """Rename columns to match the standardized schema and drop internal cols."""
        # Strip whitespace from column names (CSV headers often have trailing spaces)
        df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

        rename_map = {
            col: BVMT_COLUMN_MAP[col]
            for col in df.columns
            if col in BVMT_COLUMN_MAP
        }
        df = df.rename(columns=rename_map)

        # Drop internal columns (mapped with _ prefix — not needed downstream)
        internal_cols = [c for c in df.columns if c.startswith("_")]
        if internal_cols:
            df = df.drop(columns=internal_cols)

        # Strip whitespace from string-valued cells (CSVs have trailing spaces)
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        return df

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
