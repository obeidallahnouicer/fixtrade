"""
FixTrade — Full Training Pipeline Runner
==========================================

Runs the complete pipeline:
  1. ETL: CSV → Bronze → Silver → Gold (feature-enriched)
  2. Walk-forward CV training with MLflow tracking
  3. Final production model training
  4. Launches MLflow UI so you can compare models

Usage:
    python run_training.py              # full pipeline
    python run_training.py --skip-etl   # skip ETL (data already processed)
    python run_training.py --final-only # only train final model (skip CV)

After completion, open http://127.0.0.1:5000 to view the MLflow dashboard.
"""

import argparse
import logging
import sys
import subprocess
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_training")


def main() -> None:
    parser = argparse.ArgumentParser(description="FixTrade Training Runner")
    parser.add_argument("--skip-etl", action="store_true", help="Skip ETL, use existing Silver data")
    parser.add_argument("--final-only", action="store_true", help="Only train final model (no CV)")
    parser.add_argument("--no-ui", action="store_true", help="Don't launch MLflow UI after training")
    parser.add_argument("--port", type=int, default=5000, help="MLflow UI port (default 5000)")
    args = parser.parse_args()

    # ── Step 1: ETL ──────────────────────────────────────────────
    if not args.skip_etl:
        logger.info("=" * 70)
        logger.info("STEP 1 / 3 — ETL PIPELINE")
        logger.info("=" * 70)

        from prediction.pipeline import ETLPipeline

        etl = ETLPipeline()
        features_df = etl.run()

        if features_df.empty:
            logger.error("ETL produced no data. Check data/raw/ directory.")
            sys.exit(1)

        logger.info(
            "ETL complete: %d rows, %d columns, %d tickers",
            len(features_df),
            len(features_df.columns),
            features_df["code"].nunique() if "code" in features_df.columns else 0,
        )
    else:
        logger.info("Skipping ETL — loading existing Silver layer data...")
        from prediction.pipeline import ETLPipeline

        etl = ETLPipeline()
        features_df = etl._loader.load_layer("silver")

        if features_df.empty:
            logger.error("No Silver-layer data found. Run without --skip-etl first.")
            sys.exit(1)

        logger.info("Loaded %d rows from Silver layer.", len(features_df))

    # ── Step 2: Walk-Forward CV Training ─────────────────────────
    from prediction.training import TrainingPipeline

    trainer = TrainingPipeline()

    if not args.final_only:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2 / 3 — WALK-FORWARD CROSS-VALIDATION (MLflow tracked)")
        logger.info("=" * 70)

        avg_metrics = trainer.run(features_df)

        logger.info("")
        logger.info("=" * 70)
        logger.info("CV RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info("%-12s %10s %10s %10s %10s %10s", "Model", "MAE", "RMSE", "MAPE", "DirAcc", "R²")
        logger.info("-" * 64)
        for name, m in avg_metrics.items():
            logger.info(
                "%-12s %10.4f %10.4f %10.4f %10.4f %10.4f",
                name, m.mae, m.rmse, m.mape, m.directional_accuracy, m.r_squared,
            )

    # ── Step 3: Final Production Model ───────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3 / 3 — FINAL PRODUCTION MODEL (MLflow tracked)")
    logger.info("=" * 70)

    ensemble = trainer.train_final_model(features_df)
    logger.info("Final ensemble trained. Weights: %s", ensemble._weights)
    logger.info("Models saved to: %s", trainer._models_dir / "ensemble")

    # ── Step 4: Launch MLflow UI ─────────────────────────────────
    if not args.no_ui:
        logger.info("")
        logger.info("=" * 70)
        logger.info("LAUNCHING MLFLOW UI — http://127.0.0.1:%d", args.port)
        logger.info("=" * 70)
        logger.info("Press Ctrl+C to stop the UI server.")

        webbrowser.open(f"http://127.0.0.1:{args.port}")
        try:
            from prediction.config import config
            subprocess.run(
                [
                    sys.executable, "-m", "mlflow", "ui",
                    "--backend-store-uri", config.mlflow.tracking_uri,
                    "--port", str(args.port),
                ],
                check=True,
            )
        except KeyboardInterrupt:
            logger.info("MLflow UI stopped.")
    else:
        logger.info("")
        logger.info("Training complete. Run 'python -m prediction mlflow-ui' to view results.")


if __name__ == "__main__":
    main()
