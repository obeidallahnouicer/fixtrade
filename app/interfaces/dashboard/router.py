"""Dashboard backend-for-frontend router."""

from fastapi import APIRouter, Query

from app.interfaces.dashboard.schemas import (
    DashboardBootstrapResponse,
    MarketUniverseResponse,
)
from app.interfaces.dashboard.live_data import (
    load_persisted_dashboard,
    load_pipeline_status,
    load_market_universe,
)
from app.interfaces.dashboard.demo_data import (
    build_demo_anomalies,
    build_demo_predictions,
    build_demo_recommendation,
    build_demo_sentiment,
    load_latest_history,
)
from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/pipeline-status", summary="Get automated pipeline status")
def pipeline_status() -> dict:
    """Expose persisted ETL/scraping/prediction/anomaly job status."""
    return load_pipeline_status()


@router.get(
    "/markets",
    response_model=MarketUniverseResponse,
    summary="Get the active BVMT market universe",
)
def market_universe(
    limit: int = Query(default=60, ge=10, le=100),
) -> MarketUniverseResponse:
    """Return real market snapshots ranked by recent trading volume."""
    return MarketUniverseResponse(markets=load_market_universe(limit))


@router.get("/bootstrap", response_model=DashboardBootstrapResponse, summary="Bootstrap the dashboard")
def bootstrap_dashboard(
    symbol: str = Query(
        ...,
        min_length=2,
        max_length=50,
        pattern=r"^[A-Z0-9 .'&-]+$",
    ),
    live: bool = Query(
        True,
        description="Read persisted automated pipeline outputs before fallbacks",
    ),
) -> DashboardBootstrapResponse:
    """Aggregate the core dashboard data into one response."""

    warnings: list[str] = []
    persisted = {}
    if live:
        try:
            persisted = load_persisted_dashboard(symbol)
        except Exception:
            warnings.append("Automated pipeline data is temporarily unavailable")
    else:
        warnings.append("Demo mode requested")

    historical_prices = persisted.get("historical_prices", [])
    if not historical_prices:
        historical_prices = list(load_latest_history(symbol))
        if historical_prices:
            warnings.append("Showing the latest bundled BVMT market history")

    predictions = persisted.get("price_predictions", [])
    if not predictions:
        predictions = build_demo_predictions(symbol, historical_prices)
        if predictions:
            warnings.append("Showing a lightweight demo forecast")

    sentiment_result = persisted.get("sentiment")
    if sentiment_result is None:
        sentiment_result = build_demo_sentiment(symbol, historical_prices)

    anomaly_results = persisted.get("anomalies", [])
    if not anomaly_results:
        anomaly_results = build_demo_anomalies(symbol, historical_prices)

    recommendation_result = persisted.get("recommendation")
    if recommendation_result is None:
        recommendation_result = build_demo_recommendation(
            symbol, historical_prices, predictions
        )

    return DashboardBootstrapResponse(
        symbol=symbol,
        current_snapshot=persisted.get("current_snapshot"),
        historical_prices=[
            HistoricalPriceItem(
                date=item.date,
                close=item.close,
            )
            for item in historical_prices
        ],
        price_predictions=[
            PredictPriceItem(
                symbol=item.symbol,
                target_date=item.target_date,
                predicted_close=item.predicted_close,
                confidence_lower=item.confidence_lower,
                confidence_upper=item.confidence_upper,
            )
            for item in predictions
        ],
        sentiment=(
            SentimentResponse(
                symbol=sentiment_result.symbol,
                date=sentiment_result.date,
                score=sentiment_result.score,
                sentiment=sentiment_result.sentiment,
                article_count=sentiment_result.article_count,
            )
            if sentiment_result is not None
            else None
        ),
        anomalies=[
            AnomalyItem(
                id=item.id,
                symbol=item.symbol,
                detected_at=item.detected_at,
                anomaly_type=item.anomaly_type,
                severity=item.severity,
                description=item.description,
            )
            for item in anomaly_results
        ],
        recommendation=(
            RecommendationResponse(
                symbol=recommendation_result.symbol,
                action=recommendation_result.action,
                confidence=recommendation_result.confidence,
                reasoning=recommendation_result.reasoning,
            )
            if recommendation_result is not None
            else None
        ),
        warnings=warnings,
    )
