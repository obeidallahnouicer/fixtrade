from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="FixTrade GenAI Service", version="1.0.0")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "service": "genai-service"}

    # Expose a small wrapper endpoint for explainability
    @app.post("/api/v1/explain")
    def explain(payload: dict):
        symbol = str(payload.get("symbol", "UNKNOWN")).upper()
        action = str(payload.get("action", "hold")).lower()
        return {
            "symbol": symbol,
            "summary": (
                f"Demo explanation for {symbol}: the signal currently leans {action}. "
                f"This standalone GenAI service is isolated from the main API for the thesis demo."
            ),
            "source": "genai-service",
        }

    return app


app = create_app()
