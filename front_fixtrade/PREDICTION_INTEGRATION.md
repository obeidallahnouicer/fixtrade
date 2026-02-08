# Stock Price Prediction Integration

This document describes the integration between the front_fixtrade dashboard and the fixtrade prediction API.

## Overview

The dashboard now displays predicted stock prices alongside real-time market data. For each stock displayed, the system fetches a 1-day price prediction from the machine learning model in the fixtrade backend.

## Architecture

### Components

1. **Proxy Server** (`front_fixtrade/server/index.js`)
   - Forwards prediction requests from frontend to fixtrade backend
   - Endpoint: `POST /api/v1/trading/predictions`
   - Handles timeouts and error states gracefully

2. **API Service** (`client/src/services/api.js`)
   - `predictStockPrice(symbol, horizonDays)` - Calls the prediction endpoint
   - Returns predictions with confidence intervals

3. **Custom Hook** (`client/src/hooks/useStockPredictions.js`)
   - `useStockPrediction(symbol, horizonDays)` - React Query hook for fetching predictions
   - Caches predictions for 1 minute
   - Handles loading and error states

4. **Stock Card Component** (`client/src/components/StockCard.jsx`)
   - Displays current price with "Current" label
   - Displays predicted price with "Predicted" label and chart icon
   - Shows loading skeleton while fetching prediction
   - Gracefully handles prediction errors

## API Request/Response

### Request
```json
POST /api/v1/trading/predictions
{
  "symbol": "BNA",
  "horizon_days": 1
}
```

### Response
```json
{
  "predictions": [
    {
      "symbol": "BNA",
      "target_date": "2026-02-09",
      "predicted_close": "5.85",
      "confidence_lower": "5.75",
      "confidence_upper": "5.95"
    }
  ]
}
```

## Setup Instructions

### 1. Start the Fixtrade Backend
The prediction API must be running on port 8000:

```bash
cd back_fixtrade/fixtrade
# Make sure models are trained and available
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Configure the Proxy Server
Copy the example environment file:

```bash
cd front_fixtrade/server
cp .env.example .env
```

Verify the configuration in `.env`:
```env
FIXTRADE_API_URL=http://localhost:8000
PORT=5000
```

### 3. Start the Frontend
```bash
# Terminal 1 - Start proxy server
cd front_fixtrade/server
npm install
npm start

# Terminal 2 - Start React app
cd front_fixtrade/client
npm install
npm run dev
```

### 4. Access the Dashboard
Open http://localhost:5173 in your browser. You should see:
- Current prices for each stock
- Predicted prices displayed below current prices in blue/primary color
- Loading indicators while predictions are being fetched

## Visual Design

The predicted price is displayed with:
- **Blue/Primary color** (#6366f1) to distinguish from current price
- **Smaller font size** than current price
- **"Predicted" label** for clarity
- **Chart icon** (📈) to indicate it's a forecast
- **Loading skeleton** during fetch

## Error Handling

The system gracefully handles:
- **Backend unavailable**: Predictions simply don't show, current data remains visible
- **Invalid symbols**: Prediction request is skipped
- **Timeout**: 15-second timeout with proper error message
- **Network errors**: Logged to console, UI remains functional

## Performance Considerations

- **Caching**: Predictions are cached for 1 minute via React Query
- **Lazy loading**: Predictions load in parallel with stock data
- **Retry logic**: Failed predictions retry once after 1 second
- **Graceful degradation**: App works perfectly even if predictions fail

## Future Enhancements

- Display confidence intervals on hover
- Show multi-day predictions in detail view
- Add prediction accuracy metrics
- Enable/disable predictions via UI toggle
- Display prediction trend arrows (up/down)
