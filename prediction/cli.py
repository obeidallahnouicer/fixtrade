"""
CLI entry point for the prediction module.

Usage:
    # Run full ETL pipeline
    python -m prediction.cli etl

    # Run incremental ETL
    python -m prediction.cli etl --incremental

    # Train all models
    python -m prediction.cli train

    # Run prediction for a ticker
    python -m prediction.cli predict --symbol BIAT --days 5

    # Warm the prediction cache
    python -m prediction.cli warm-cache
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_etl(args: argparse.Namespace) -> None:
    """Run the ETL pipeline."""
    from prediction.pipeline import ETLPipeline

    pipeline = ETLPipeline()
    if args.incremental:
        result = pipeline.run_incremental()
    else:
        result = pipeline.run()

    if result.empty:
        logger.warning("ETL produced no data.")
    else:
        logger.info("ETL complete: %d rows.", len(result))


def cmd_train(args: argparse.Namespace) -> None:
    """Train ML models, optionally for a single symbol."""
    from prediction.pipeline import ETLPipeline
    from prediction.training import TrainingPipeline

    # Load features from Silver layer
    etl = ETLPipeline()
    features_df = etl._loader.load_layer("silver")
    if features_df.empty:
        logger.error("No Silver-layer data. Run ETL first: python -m prediction etl")
        sys.exit(1)

    # Filter to a single symbol if requested
    symbol = getattr(args, "symbol", None)
    if symbol:
        if "libelle" in features_df.columns:
            features_df = features_df[
                features_df["libelle"].str.upper() == symbol.upper()
            ]
        elif "code" in features_df.columns:
            features_df = features_df[features_df["code"] == symbol]
        if features_df.empty:
            logger.error("No data found for symbol '%s'.", symbol)
            sys.exit(1)
        logger.info("Training on symbol %s — %d rows.", symbol, len(features_df))

    trainer = TrainingPipeline()

    if args.final:
        ensemble = trainer.train_final_model(features_df, symbol=symbol)
        logger.info("Final model trained and saved. Ensemble: %s", ensemble.name)
    else:
        metrics = trainer.run(features_df, symbol=symbol)
        for name, m in metrics.items():
            logger.info("[%s] %s", name, m)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run a single prediction."""
    from prediction.inference import PredictionService

    service = PredictionService()
    results = service.predict(
        symbol=args.symbol,
        horizon_days=args.days,
        model=args.model,
    )

    for r in results:
        logger.info(
            "%s | %s | Close=%.3f | CI=[%.3f, %.3f] | Model=%s",
            r.symbol,
            r.target_date,
            r.predicted_close,
            r.confidence_lower,
            r.confidence_upper,
            r.model_name,
        )


def cmd_warm_cache(args: argparse.Namespace) -> None:
    """Pre-compute predictions for top tickers."""
    from prediction.config import config
    from prediction.inference import PredictionService
    from prediction.utils.cache import CacheClient

    service = PredictionService()
    cache = CacheClient(redis_url=config.redis.url)

    def prediction_fn(ticker: str) -> dict:
        results = service.predict(ticker, horizon_days=5)
        return service._serialize_predictions(results)

    warmed = cache.warm_cache(
        list(config.tracked_tickers[:30]),
        prediction_fn,
    )
    logger.info("Cache warmed for %d tickers.", warmed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FixTrade Prediction Module CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ETL
    etl_parser = subparsers.add_parser("etl", help="Run ETL pipeline")
    etl_parser.add_argument(
        "--incremental", action="store_true",
        help="Only process new data since last watermark",
    )
    etl_parser.set_defaults(func=cmd_etl)

    # Train
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument(
        "--symbol", default=None,
        help="Train only for this ticker symbol (e.g. BIAT)",
    )
    train_parser.add_argument(
        "--final", action="store_true",
        help="Train the final production model (saves to registry)",
    )
    train_parser.set_defaults(func=cmd_train)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--symbol", required=True, help="Ticker symbol")
    predict_parser.add_argument("--days", type=int, default=1, help="Horizon days (1-5)")
    predict_parser.add_argument("--model", default="ensemble", help="Model name")
    predict_parser.set_defaults(func=cmd_predict)

    # Warm cache
    warm_parser = subparsers.add_parser("warm-cache", help="Pre-compute top ticker predictions")
    warm_parser.set_defaults(func=cmd_warm_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
