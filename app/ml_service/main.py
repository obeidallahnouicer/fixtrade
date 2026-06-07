"""Standalone FastAPI service for ML inference."""

from contextlib import asynccontextmanager
from datetime import date, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request

from app.ml_service.schemas import (
    HealthResponse,
    LiquidityItem,
    LiquidityResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
    VolumeItem,
    VolumeResponse,
)

class DemoPredictionService:
    def predict(self, symbol: str, horizon_days: int):
        base_price = Decimal("100.000")
        results = []
        for offset in range(1, horizon_days + 1):
            target_date = date.today() + timedelta(days=offset)
            price = base_price + Decimal(str(offset * 1.25))
            results.append(
                type(
                    "Prediction",
                    (),
                    {
                        "symbol": symbol,
                        "target_date": target_date,
                        "predicted_close": price,
                        "confidence_lower": price - Decimal("2.500"),
                        "confidence_upper": price + Decimal("2.500"),
                    },
                )()
            )
        return results

    def predict_volume(self, symbol: str, horizon_days: int):
        results = []
        for offset in range(1, horizon_days + 1):
            target_date = date.today() + timedelta(days=offset)
            results.append(
                type(
                    "VolumePrediction",
                    (),
                    {
                        "symbol": symbol,
                        "target_date": target_date,
                        "predicted_volume": 10000 + offset * 500,
                    },
                )()
            )
        return results

    def predict_liquidity(self, symbol: str, horizon_days: int):
        results = []
        for offset in range(1, horizon_days + 1):
            target_date = date.today() + timedelta(days=offset)
            results.append(
                type(
                    "LiquidityPrediction",
                    (),
                    {
                        "symbol": symbol,
                        "target_date": target_date,
                        "prob_low": Decimal("0.10"),
                        "prob_medium": Decimal("0.25"),
                        "prob_high": Decimal("0.65"),
                        "predicted_tier": "high",
                    },
                )()
            )
        return results


def _get_prediction_service(app: FastAPI):
    service = getattr(app.state, "prediction_service", None)
    if service is not None:
        return service

    try:
        from prediction.inference import PredictionService

        service = PredictionService()
    except Exception:
        service = DemoPredictionService()

    app.state.prediction_service = service
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = None
    app.state.ready = True
    yield


app = FastAPI(title="FixTrade ML Service", version="1.0.0", lifespan=lifespan)


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = "ok" if getattr(app.state, "ready", False) else "starting"
    return HealthResponse(status=status, service="ml-service")


@app.post("/api/v1/predictions", response_model=PredictionResponse)
def predict_prices(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service = _get_prediction_service(request.app)
    results = service.predict(payload.symbol, payload.horizon_days)
    return PredictionResponse(
        predictions=[
            PredictionItem(
                symbol=item.symbol,
                target_date=item.target_date,
                predicted_close=item.predicted_close,
                confidence_lower=item.confidence_lower,
                confidence_upper=item.confidence_upper,
            )
            for item in results
        ]
    )


@app.post("/api/v1/predictions/volume", response_model=VolumeResponse)
def predict_volume(payload: PredictionRequest, request: Request) -> VolumeResponse:
    service = _get_prediction_service(request.app)
    results = service.predict_volume(payload.symbol, payload.horizon_days)
    return VolumeResponse(
        predictions=[
            VolumeItem(
                symbol=item.symbol,
                target_date=item.target_date,
                predicted_volume=int(item.predicted_volume),
            )
            for item in results
        ]
    )


@app.post("/api/v1/predictions/liquidity", response_model=LiquidityResponse)
def predict_liquidity(payload: PredictionRequest, request: Request) -> LiquidityResponse:
    service = _get_prediction_service(request.app)
    results = service.predict_liquidity(payload.symbol, payload.horizon_days)
    return LiquidityResponse(
        predictions=[
            LiquidityItem(
                symbol=item.symbol,
                target_date=item.target_date,
                prob_low=item.prob_low,
                prob_medium=item.prob_medium,
                prob_high=item.prob_high,
                predicted_tier=item.predicted_tier,
            )
            for item in results
        ]
    )