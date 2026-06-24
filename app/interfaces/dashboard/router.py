"""Dashboard backend-for-frontend router."""

from datetime import date, timedelta
from uuid import UUID

from fastapi import APIRouter, Query

from app.application.trading.dtos import (
    DetectAnomaliesQuery,
    GetRecommendationQuery,
    GetSentimentQuery,
    PredictPriceCommand,
)
from app.domain.trading.errors import PortfolioNotFoundError
from app.interfaces.dashboard.schemas import DashboardBootstrapResponse
from app.interfaces.dashboard.demo_data import (
    build_demo_anomalies,
    build_demo_predictions,
    build_demo_recommendation,
    build_demo_sentiment,
    load_latest_history,
)
from app.interfaces.trading.dependencies import (
    get_detect_anomalies_use_case,
    get_predict_price_use_case,
    get_stock_price_repository,
    get_recommendation_use_case,
    get_sentiment_use_case,
)
from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
DEFAULT_PORTFOLIO_ID = UUID("00000000-0000-0000-0000-000000000000")


@router.get("/bootstrap", response_model=DashboardBootstrapResponse, summary="Bootstrap the dashboard")
def bootstrap_dashboard(
    symbol: str = Query(..., min_length=2, max_length=10, pattern=r"^[A-Z0-9]+$"),
    live: bool = Query(
        False,
        description="Try live database and ML integrations before demo fallbacks",
    ),
) -> DashboardBootstrapResponse:
    """Aggregate the core dashboard data into one response."""

    warnings: list[str] = []
    if not live:
        warnings.append("Demo mode: using bundled BVMT data and local signals")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    historical_prices = []
    if live:
        try:
            price_repo = get_stock_price_repository()
            historical_prices = price_repo.get_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
            )
        except Exception:
            warnings.append("Historical price data temporarily unavailable")
    if not historical_prices:
        historical_prices = list(load_latest_history(symbol))
        if historical_prices:
            warnings.append("Showing the latest bundled BVMT market history")

    predictions = []
    if live:
        try:
            predict_use_case = get_predict_price_use_case()
            predictions = predict_use_case.execute(
                PredictPriceCommand(symbol=symbol, horizon_days=5)
            )
        except Exception:
            warnings.append("Price predictions temporarily unavailable")
    if not predictions:
        predictions = build_demo_predictions(symbol, historical_prices)
        if predictions:
            warnings.append("Showing a lightweight demo forecast")

    sentiment_result = None
    if live:
        try:
            sentiment_use_case = get_sentiment_use_case()
            sentiment_result = sentiment_use_case.execute(
                GetSentimentQuery(symbol=symbol, target_date=None)
            )
        except Exception:
            warnings.append("Sentiment signal temporarily unavailable")
    if sentiment_result is None:
        sentiment_result = build_demo_sentiment(symbol, historical_prices)

    anomaly_results = []
    if live:
        try:
            anomaly_use_case = get_detect_anomalies_use_case()
            anomaly_results = anomaly_use_case.execute(
                DetectAnomaliesQuery(symbol=symbol)
            )
        except Exception:
            warnings.append("Anomaly detection temporarily unavailable")
    if not anomaly_results:
        anomaly_results = build_demo_anomalies(symbol, historical_prices)

    recommendation_result = None
    if live:
        try:
            recommendation_use_case = get_recommendation_use_case()
            recommendation_result = recommendation_use_case.execute(
                GetRecommendationQuery(
                    symbol=symbol, portfolio_id=DEFAULT_PORTFOLIO_ID
                )
            )
        except PortfolioNotFoundError:
            warnings.append("Default portfolio not found; recommendation unavailable")
        except Exception:
            warnings.append("Recommendation temporarily unavailable")
    if recommendation_result is None:
        recommendation_result = build_demo_recommendation(
            symbol, historical_prices, predictions
        )

    return DashboardBootstrapResponse(
        symbol=symbol,
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
