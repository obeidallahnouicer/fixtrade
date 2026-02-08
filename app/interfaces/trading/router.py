"""
FastAPI router for the trading bounded context.

All routes delegate to use cases. No business logic here.
Input validation is handled by Pydantic schemas.
Error mapping is handled by centralized error handlers.
"""

from fastapi import APIRouter, Depends

from app.application.trading.dtos import (
    DetectAnomaliesQuery,
    GetRecommendationQuery,
    GetSentimentQuery,
    PredictPriceCommand,
)
from app.application.trading.detect_anomalies import DetectAnomaliesUseCase
from app.application.trading.get_recommendation import GetRecommendationUseCase
from app.application.trading.get_sentiment import GetSentimentUseCase
from app.application.trading.predict_price import PredictPriceUseCase
from app.interfaces.trading.dependencies import (
    get_detect_anomalies_use_case,
    get_predict_price_use_case,
    get_recommendation_use_case,
    get_sentiment_use_case,
)
from app.interfaces.trading.schemas import (
    AnomalyItem,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    ErrorResponse,
    GetRecommendationRequest,
    GetSentimentRequest,
    PredictPriceItem,
    PredictPriceRequest,
    PredictPriceResponse,
    RecommendationResponse,
    SentimentResponse,
)

router = APIRouter(prefix="/trading", tags=["trading"])


@router.post(
    "/predictions",
    response_model=PredictPriceResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Predict stock prices",
    description="Predict closing prices for a BVMT stock over 1-5 trading days.",
)
def predict_price(
    request: PredictPriceRequest,
    use_case: PredictPriceUseCase = Depends(get_predict_price_use_case),
) -> PredictPriceResponse:
    """Predict stock prices for a given symbol and horizon."""
    command = PredictPriceCommand(
        symbol=request.symbol,
        horizon_days=request.horizon_days,
    )
    results = use_case.execute(command)
    return PredictPriceResponse(
        predictions=[
            PredictPriceItem(
                symbol=r.symbol,
                target_date=r.target_date,
                predicted_close=r.predicted_close,
                confidence_lower=r.confidence_lower,
                confidence_upper=r.confidence_upper,
            )
            for r in results
        ]
    )


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Get sentiment analysis",
    description="Retrieve aggregated sentiment for a BVMT stock symbol.",
)
def get_sentiment(
    request: GetSentimentRequest,
    use_case: GetSentimentUseCase = Depends(get_sentiment_use_case),
) -> SentimentResponse:
    """Get sentiment analysis for a given symbol."""
    query = GetSentimentQuery(
        symbol=request.symbol,
        target_date=request.target_date,
    )
    result = use_case.execute(query)
    return SentimentResponse(
        symbol=result.symbol,
        date=result.date,
        score=result.score,
        sentiment=result.sentiment,
        article_count=result.article_count,
    )


@router.post(
    "/anomalies",
    response_model=DetectAnomaliesResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Detect market anomalies",
    description="Detect anomalies in market data for a BVMT stock symbol.",
)
def detect_anomalies(
    request: DetectAnomaliesRequest,
    use_case: DetectAnomaliesUseCase = Depends(get_detect_anomalies_use_case),
) -> DetectAnomaliesResponse:
    """Detect anomalies for a given symbol."""
    query = DetectAnomaliesQuery(symbol=request.symbol)
    results = use_case.execute(query)
    return DetectAnomaliesResponse(
        anomalies=[
            AnomalyItem(
                id=r.id,
                symbol=r.symbol,
                anomaly_type=r.anomaly_type,
                severity=r.severity,
                description=r.description,
            )
            for r in results
        ]
    )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Get trade recommendation",
    description="Get a buy/sell/hold recommendation for a BVMT stock.",
)
def get_recommendation(
    request: GetRecommendationRequest,
    use_case: GetRecommendationUseCase = Depends(get_recommendation_use_case),
) -> RecommendationResponse:
    """Get a trade recommendation for a given symbol and portfolio."""
    query = GetRecommendationQuery(
        symbol=request.symbol,
        portfolio_id=request.portfolio_id,
    )
    result = use_case.execute(query)
    return RecommendationResponse(
        symbol=result.symbol,
        action=result.action,
        confidence=result.confidence,
        reasoning=result.reasoning,
    )
