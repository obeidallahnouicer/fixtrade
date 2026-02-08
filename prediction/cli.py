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

    # --top-n: keep only the N tickers with the most rows (most liquid)
    if args.top_n and "code" in features_df.columns:
        counts = features_df["code"].value_counts()
        top_tickers = counts.head(args.top_n).index.tolist()
        features_df = features_df[features_df["code"].isin(top_tickers)]
        logger.info(
            "Filtered to top %d tickers (%d rows): %s",
            args.top_n, len(features_df), top_tickers[:5],
        )

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


def cmd_predict_volume(args: argparse.Namespace) -> None:
    """Predict daily transaction volume for the next N trading days."""
    from prediction.inference import PredictionService

    service = PredictionService()
    results = service.predict_volume(
        symbol=args.symbol,
        horizon_days=args.days,
    )

    if not results:
        logger.warning("No volume predictions for %s.", args.symbol)
        return

    for r in results:
        logger.info(
            "%s | %s | Volume=%.0f",
            r.symbol,
            r.target_date,
            r.predicted_volume,
        )


def cmd_predict_liquidity(args: argparse.Namespace) -> None:
    """Predict liquidity tier probabilities for the next N trading days."""
    from prediction.inference import PredictionService

    service = PredictionService()
    results = service.predict_liquidity(
        symbol=args.symbol,
        horizon_days=args.days,
    )

    if not results:
        logger.warning("No liquidity predictions for %s.", args.symbol)
        return

    for r in results:
        logger.info(
            "%s | %s | P(low)=%.4f  P(med)=%.4f  P(high)=%.4f  → %s",
            r.symbol,
            r.target_date,
            r.prob_low,
            r.prob_medium,
            r.prob_high,
            r.predicted_tier,
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


def cmd_mlflow_ui(args: argparse.Namespace) -> None:
    """Launch the MLflow tracking UI."""
    import subprocess
    from prediction.config import config

    tracking_uri = config.mlflow.tracking_uri
    port = args.port
    logger.info("Starting MLflow UI at http://127.0.0.1:%d", port)
    logger.info("Tracking URI: %s", tracking_uri)
    subprocess.run(
        [
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri,
            "--port", str(port),
        ],
        check=True,
    )


def cmd_scheduler(args: argparse.Namespace) -> None:
    """Start the real-time scheduler (runs in foreground)."""
    from prediction.realtime.scheduler import RealtimeScheduler
    from prediction.realtime.stream import PredictionStreamManager

    stream = PredictionStreamManager()
    scheduler = RealtimeScheduler(
        stream_manager=stream,
        top_n_tickers=args.top_n,
    )
    scheduler.start()
    logger.info("Scheduler running. Press Ctrl+C to stop.")

    if args.run:
        # Execute a single task immediately and exit
        result = scheduler.run_now(args.run)
        logger.info(
            "Task '%s' %s (%.1fs)",
            result.task_name, result.status.value, result.duration_seconds,
        )
        if result.error:
            logger.error("Error: %s", result.error)
        scheduler.stop()
        return

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
    finally:
        scheduler.stop()


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch data/raw/ for new CSV files and trigger pipeline."""
    from prediction.realtime.scheduler import RealtimeScheduler
    from prediction.realtime.stream import PredictionStreamManager
    from prediction.realtime.watcher import DataWatcher

    stream = PredictionStreamManager()
    scheduler = RealtimeScheduler(
        stream_manager=stream,
        top_n_tickers=args.top_n,
    )
    watcher = DataWatcher(
        poll_interval=args.interval,
        auto_retrain=args.auto_retrain,
        scheduler=scheduler,
        stream_manager=stream,
    )

    scheduler.start()
    watcher.start()
    logger.info(
        "Watching %s (every %.0fs, auto_retrain=%s). Press Ctrl+C to stop.",
        watcher._watch_dir, args.interval, args.auto_retrain,
    )

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
    finally:
        watcher.stop()
        scheduler.stop()


def cmd_stream(args: argparse.Namespace) -> None:
    """Start the WebSocket/SSE prediction stream server.

    Two modes:
    - ``--standalone`` (default): lightweight server with only the
      realtime endpoints. No dependency on app.core.config / slowapi.
    - ``--full``: start the complete FastAPI application including
      trading, health, and realtime routers.
    """
    import uvicorn

    if args.full:
        # Full app — requires all dependencies (slowapi, pydantic-settings, …)
        logger.info("Starting FULL application at http://0.0.0.0:%d", args.port)
        uvicorn.run("app.main:app", host="0.0.0.0", port=args.port, reload=False)
    else:
        # Standalone lightweight server — only realtime endpoints
        logger.info(
            "Starting standalone realtime server at http://0.0.0.0:%d", args.port
        )
        logger.info(
            "WebSocket: ws://0.0.0.0:%d/ws/predictions", args.port
        )
        logger.info(
            "SSE:       http://0.0.0.0:%d/stream/predictions", args.port
        )
        logger.info("Docs:      http://0.0.0.0:%d/docs", args.port)

        from prediction.realtime.server import create_standalone_app

        app = create_standalone_app(top_n=args.top_n)
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)


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
    train_parser.add_argument(
        "--top-n", type=int, default=None, dest="top_n",
        help="Train only on the top N most-traded tickers (speeds up training)",
    )
    train_parser.set_defaults(func=cmd_train)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--symbol", required=True, help="Ticker symbol")
    predict_parser.add_argument("--days", type=int, default=1, help="Horizon days (1-5)")
    predict_parser.add_argument("--model", default="ensemble", help="Model name")
    predict_parser.set_defaults(func=cmd_predict)

    # Predict Volume
    vol_parser = subparsers.add_parser(
        "predict-volume", help="Predict daily transaction volume"
    )
    vol_parser.add_argument("--symbol", required=True, help="Ticker symbol")
    vol_parser.add_argument(
        "--days", type=int, default=5, help="Horizon days (1-5)"
    )
    vol_parser.set_defaults(func=cmd_predict_volume)

    # Predict Liquidity
    liq_parser = subparsers.add_parser(
        "predict-liquidity", help="Predict liquidity tier probabilities"
    )
    liq_parser.add_argument("--symbol", required=True, help="Ticker symbol")
    liq_parser.add_argument(
        "--days", type=int, default=5, help="Horizon days (1-5)"
    )
    liq_parser.set_defaults(func=cmd_predict_liquidity)

    # Warm cache
    warm_parser = subparsers.add_parser("warm-cache", help="Pre-compute top ticker predictions")
    warm_parser.set_defaults(func=cmd_warm_cache)

    # MLflow UI
    mlflow_parser = subparsers.add_parser("mlflow-ui", help="Launch MLflow tracking dashboard")
    mlflow_parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI (default 5000)")
    mlflow_parser.set_defaults(func=cmd_mlflow_ui)

    # Scheduler
    sched_parser = subparsers.add_parser(
        "scheduler", help="Start the real-time scheduler"
    )
    sched_parser.add_argument(
        "--top-n", type=int, default=10, dest="top_n",
        help="Number of top tickers to manage (default 10)",
    )
    sched_parser.add_argument(
        "--run", type=str, default=None,
        help="Run a single task and exit: etl, retrain, warm_cache, "
             "refresh_predictions, full_pipeline",
    )
    sched_parser.set_defaults(func=cmd_scheduler)

    # Watch
    watch_parser = subparsers.add_parser(
        "watch", help="Watch data/raw/ for new CSVs → auto ETL"
    )
    watch_parser.add_argument(
        "--interval", type=float, default=30.0,
        help="Seconds between directory scans (default 30)",
    )
    watch_parser.add_argument(
        "--auto-retrain", action="store_true", dest="auto_retrain",
        help="Automatically retrain models when new data is detected",
    )
    watch_parser.add_argument(
        "--top-n", type=int, default=10, dest="top_n",
        help="Number of top tickers to manage (default 10)",
    )
    watch_parser.set_defaults(func=cmd_watch)

    # Stream server
    stream_parser = subparsers.add_parser(
        "stream", help="Start the WebSocket/SSE prediction stream server"
    )
    stream_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the stream server (default 8000)",
    )
    stream_parser.add_argument(
        "--full", action="store_true",
        help="Start the full FastAPI app (requires all deps like slowapi)",
    )
    stream_parser.add_argument(
        "--top-n", type=int, default=10, dest="top_n",
        help="Number of top tickers to manage (default 10)",
    )
    stream_parser.set_defaults(func=cmd_stream)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
