"""
Multi-ticker walk-forward CV training for FixTrade.

Trains on all tickers that have at least MIN_ROWS rows in the Silver layer,
saves per-ticker metrics to logs/metrics_all_tickers.json, and prints a
summary table suitable for the thesis.

Usage (inside Docker):
    python scripts/train_all_tickers.py
    python scripts/train_all_tickers.py --top-n 15
    python scripts/train_all_tickers.py --min-rows 500
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_ROOT = Path(__file__).parent.parent.absolute()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.absolute()
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_all_tickers")

MIN_ROWS_DEFAULT = 400
OUT_FILE = Path("logs/metrics_all_tickers.json")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT)
    args = parser.parse_args()

    from prediction.pipeline import ETLPipeline
    from prediction.training import TrainingPipeline

    # Load Silver layer
    etl = ETLPipeline()
    silver_df = etl._loader.load_layer("silver")
    if silver_df.empty:
        logger.error("Silver layer empty. Run ETL first.")
        sys.exit(1)

    logger.info("Silver layer: %d rows", len(silver_df))

    # Discover available tickers — prefer human-readable libelle
    ticker_col = "libelle" if "libelle" in silver_df.columns else "code"
    counts = silver_df[ticker_col].value_counts()
    eligible = counts[counts >= args.min_rows]

    if args.top_n:
        eligible = eligible.head(args.top_n)

    tickers = eligible.index.tolist()
    logger.info(
        "Will train on %d tickers (min_rows=%d): %s",
        len(tickers), args.min_rows, tickers,
    )

    results = {}

    for i, ticker in enumerate(tickers, 1):
        logger.info(
            "=" * 70
        )
        logger.info(
            "[%d/%d] Training ticker: %s  (%d rows)",
            i, len(tickers), ticker, counts[ticker],
        )
        logger.info("=" * 70)

        df_ticker = silver_df[silver_df[ticker_col] == ticker].copy()

        try:
            trainer = TrainingPipeline()
            cv_metrics = trainer.run(df_ticker)

            ticker_result = {
                "n_rows": int(counts[ticker]),
                "models": {},
            }
            for model_name, m in cv_metrics.items():
                ticker_result["models"][model_name] = {
                    "mae":    round(m.mae, 4),
                    "rmse":   round(m.rmse, 4),
                    "mape":   round(m.mape * 100, 4),  # store as %
                    "dir_acc": round(m.directional_accuracy * 100, 2),
                    "r2":     round(m.r_squared, 4),
                }
                logger.info(
                    "[%s][%s] MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  DirAcc=%.2f%%  R2=%.4f",
                    ticker, model_name,
                    m.mae, m.rmse, m.mape * 100,
                    m.directional_accuracy * 100, m.r_squared,
                )

            results[ticker] = ticker_result

        except Exception:
            logger.exception("Failed to train %s", ticker)
            results[ticker] = {"n_rows": int(counts[ticker]), "error": "training_failed"}

        # Save after every ticker so we don't lose progress on crash
        OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print summary table
    logger.info("\n%s", "=" * 90)
    logger.info("SUMMARY — Walk-forward CV metrics (averaged across splits)")
    logger.info("=" * 90)
    header = f"{'Ticker':<12} {'Rows':>6}  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'DirAcc%':>9} {'R2':>8}"
    logger.info(header)
    logger.info("-" * 90)

    for ticker, res in results.items():
        if "error" in res:
            logger.info("%-12s %6d  ERROR: %s", ticker, res["n_rows"], res["error"])
            continue
        for model_name, m in res["models"].items():
            logger.info(
                "%-12s %6d  %-14s %8.4f %8.4f %8.2f %9.2f %8.4f",
                ticker, res["n_rows"], model_name,
                m["mae"], m["rmse"], m["mape"], m["dir_acc"], m["r2"],
            )

    logger.info("\nMetrics saved to %s", OUT_FILE.resolve())


if __name__ == "__main__":
    main()
