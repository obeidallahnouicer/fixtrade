"""
ETL sub-package — Extract → Transform → Load.

Layers
------
- **Extract** : `BVMTExtractor`   — CSV ingestion, encoding detection, dedup
- **Bronze → Silver** : `BronzeToSilverTransformer` — validation, type coercion, null fill
- **Silver → Gold** : `SilverToGoldTransformer` — target creation, chrono splits
- **Load** : `ParquetLoader` — partitioned Parquet I/O, watermark tracking
"""

from prediction.etl.extract.bvmt_extractor import BVMTExtractor
from prediction.etl.transform.bronze_to_silver import (
    BronzeToSilverTransformer,
    DataQualityChecker,
)
from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
from prediction.etl.load.parquet_loader import ParquetLoader

__all__ = [
    "BVMTExtractor",
    "BronzeToSilverTransformer",
    "DataQualityChecker",
    "SilverToGoldTransformer",
    "ParquetLoader",
]
