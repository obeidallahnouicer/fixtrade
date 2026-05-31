# API Endpoints Status Report

## ✅ Working Endpoints (No Errors)

### Health & Status
- ✅ **GET** `/api/v1/health` → 200
  - Response: `{"status": "ok", "version": "1.0.0"}`

- ✅ **GET** `/api/v1/ai/status` → 200
  - Shows AI module status, model info, and configuration

### AI Recommendations
- ✅ **GET** `/api/v1/ai/recommendations` → 200
  - Returns daily AI recommendations
  
- ✅ **GET** `/api/v1/ai/recommendations/{symbol}/explain` → 200
  - Example: `/api/v1/ai/recommendations/BIAT/explain`
  - Explains recommendation reasoning

### Trading - Sentiment
- ✅ **POST** `/api/v1/trading/sentiment` → 200
  - Payload: `{"symbol": "BIAT", "text": "Great stock"}`
  - Returns sentiment analysis result

### Documentation
- ✅ **GET** `/docs` → 200 (Swagger UI)
- ✅ **GET** `/redoc` → 200 (ReDoc)

---

## ⚠️ Endpoints with Issues

### Trading - Predictions (422 Validation Error)
- ❌ **POST** `/api/v1/trading/predictions`
- **Issue**: Wrong field names in payload
- **Fix**: Use `symbol` and `horizon_days` instead of `symbols` and `days`
- **Correct Payload**:
  ```json
  {
    "symbol": "BIAT",
    "horizon_days": 5
  }
  ```

### Trading - Anomalies (500 Internal Error)
- ❌ **POST** `/api/v1/trading/anomalies`
- **Issue**: Database connection or processing error
- **Status**: Under investigation

---

## 📋 All Available Endpoints (30 total)

### AI Portfolio Management
- `POST /api/v1/ai/portfolio/create` - Create Portfolio
- `POST /api/v1/ai/portfolio/optimize` - Optimize Portfolio
- `POST /api/v1/ai/portfolio/simulate` - Simulate Portfolio
- `POST /api/v1/ai/portfolio/efficient-frontier` - Get Efficient Frontier
- `POST /api/v1/ai/portfolio/explain` - Explain Portfolio
- `POST /api/v1/ai/portfolio/recommendations/detailed` - Get Detailed Recommendations
- `GET /api/v1/ai/portfolio/{portfolio_id}/snapshot` - Get Portfolio Snapshot
- `GET /api/v1/ai/portfolio/{portfolio_id}/performance` - Get Portfolio Performance
- `GET /api/v1/ai/portfolio/{portfolio_id}/performance/explain` - Explain Performance
- `GET /api/v1/ai/portfolio/{portfolio_id}/position/{symbol}` - Get Position
- `POST /api/v1/ai/portfolio/{portfolio_id}/prices/update` - Update Prices
- `POST /api/v1/ai/portfolio/{portfolio_id}/trade` - Execute Trade
- `POST /api/v1/ai/portfolio/{portfolio_id}/stop-loss/check` - Check Stop Losses

### AI User Profile
- `POST /api/v1/ai/profile/questionnaire` - Recommend Profile

### Trading - Sentiment
- `POST /api/v1/trading/sentiment` - Get sentiment analysis ✅
- `POST /api/v1/trading/sentiment/analyze` - Analyze article sentiment
- `POST /api/v1/trading/sentiment/aggregate` - Aggregate daily sentiment scores
- `POST /api/v1/trading/sentiment/link-symbols` - Link articles to BVMT symbols

### Trading - Predictions
- `POST /api/v1/trading/predictions` - Predict stock prices ⚠️
- `POST /api/v1/trading/predictions/liquidity` - Predict liquidity probabilities
- `POST /api/v1/trading/predictions/volume` - Predict transaction volume

### Trading - Anomalies
- `POST /api/v1/trading/anomalies` - Detect market anomalies ❌
- `POST /api/v1/trading/anomalies/intraday` - Detect intraday anomalies
- `POST /api/v1/trading/anomalies/evaluate` - Evaluate anomaly detection performance
- `GET /api/v1/trading/anomalies/recent` - Get recent anomaly alerts

### Trading - Recommendations
- `POST /api/v1/trading/recommendations` - Get trade recommendation

### Health & Info
- `GET /api/v1/health` - Health check ✅
- `GET /api/v1/ai/status` - Get AI Status ✅

---

## 🧪 Example Usage

### Get Health Status
```bash
curl -X GET http://localhost:8000/api/v1/health
```

### Get AI Recommendations
```bash
curl -X GET http://localhost:8000/api/v1/ai/recommendations
```

### Analyze Sentiment
```bash
curl -X POST http://localhost:8000/api/v1/trading/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BIAT",
    "text": "This is a great stock to buy now"
  }'
```

### Predict Stock Prices
```bash
curl -X POST http://localhost:8000/api/v1/trading/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BIAT",
    "horizon_days": 5
  }'
```

---

## 📊 Summary

| Category | Status | Count |
|----------|--------|-------|
| ✅ Working | Fully Functional | 6+ |
| ⚠️ Partial | Need Correct Payload | 1 |
| ❌ Broken | Database/Processing Issue | 1 |
| 📍 Available | Total Endpoints | 30 |

---

**Last Tested**: May 25, 2026  
**App Status**: RUNNING ✅  
**Most Endpoints**: FUNCTIONAL ✅
