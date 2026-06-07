"""Dashboard backend-for-frontend router."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.application.trading.dtos import (
    DetectAnomaliesQuery,
    GetRecommendationQuery,
    GetSentimentQuery,
    PredictPriceCommand,
)
from app.domain.trading.errors import PortfolioNotFoundError
from app.interfaces.dashboard.schemas import DashboardBootstrapResponse
from app.interfaces.trading.dependencies import (
    get_detect_anomalies_use_case,
    get_predict_price_use_case,
    get_recommendation_use_case,
    get_sentiment_use_case,
)
from app.interfaces.trading.schemas import (
    AnomalyItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
DEFAULT_PORTFOLIO_ID = UUID("00000000-0000-0000-0000-000000000000")


@router.get("/bootstrap", response_model=DashboardBootstrapResponse, summary="Bootstrap the dashboard")
def bootstrap_dashboard(
    symbol: str = Query(..., min_length=2, max_length=10, pattern=r"^[A-Z0-9]+$"),
    predict_use_case=Depends(get_predict_price_use_case),
    sentiment_use_case=Depends(get_sentiment_use_case),
    anomaly_use_case=Depends(get_detect_anomalies_use_case),
    recommendation_use_case=Depends(get_recommendation_use_case),
) -> DashboardBootstrapResponse:
    """Aggregate the core dashboard data into one response."""

    warnings: list[str] = []

    predictions = []
    try:
        predictions = predict_use_case.execute(
            PredictPriceCommand(symbol=symbol, horizon_days=5)
        )
    except Exception:
        warnings.append("Price predictions temporarily unavailable")

    sentiment_result = None
    try:
        sentiment_result = sentiment_use_case.execute(
            GetSentimentQuery(symbol=symbol, target_date=None)
        )
    except Exception:
        warnings.append("Sentiment signal temporarily unavailable")

    anomaly_results = []
    try:
        anomaly_results = anomaly_use_case.execute(DetectAnomaliesQuery(symbol=symbol))
    except Exception:
        warnings.append("Anomaly detection temporarily unavailable")

    recommendation_result = None
    try:
        recommendation_result = recommendation_use_case.execute(
            GetRecommendationQuery(symbol=symbol, portfolio_id=DEFAULT_PORTFOLIO_ID)
        )
    except PortfolioNotFoundError:
        warnings.append("Default portfolio not found; recommendation unavailable")
    except Exception:
        warnings.append("Recommendation temporarily unavailable")

    return DashboardBootstrapResponse(
        symbol=symbol,
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