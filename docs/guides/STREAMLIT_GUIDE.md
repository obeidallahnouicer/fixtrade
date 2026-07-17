# FixTrade Streamlit Dashboard — Quick Start Guide

## 🚀 Launch Instructions

### 1. Start the Backend (FastAPI)
```bash
cd /home/obeid/Desktop/projects/fixtrade
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend (Streamlit)
In a **new terminal**:
```bash
cd /home/obeid/Desktop/projects/fixtrade
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

### 3. Access the Dashboard
Open your browser: **http://localhost:8501**

---

## 🎯 NEW: Smart Trading Page

The **Smart Trading (NEW)** page is your all-in-one trading intelligence hub:

### Tab 1: 📊 Price Analysis
- **Historical prices** (last 90 days of real data)
- **Predicted prices** (5-day forecast with confidence intervals)
- **Interactive chart** showing historical + predicted prices
- **Trading signal** (BUY/SELL/HOLD based on price momentum)
- **Real-time anomaly check**
- **Data tables** for historical and predicted prices

### Tab 2: 🎯 Portfolio Recommendations
- **CAPM-based recommendations** for your portfolio
- Select multiple symbols for diversified analysis
- Choose risk profile: Conservative / Moderate / Aggressive
- **Optional AI explanations** using Groq/OpenAI LLMs
- Shows:
  - BUY/SELL/HOLD signals with confidence scores
  - Expected returns and Beta coefficients
  - Current vs Target portfolio weights
  - Anomaly detection per symbol
  - Natural language explanations

### Tab 3: 📈 Full Report
Complete analysis for a single stock:
- **Price Forecast** (5-day predictions)
- **Sentiment Analysis** (news sentiment score)
- **Anomaly Alerts** (unusual market activity)
- **Volume Forecast** (expected trading volume)
- **Liquidity Forecast** (liquidity tier probabilities)

---

## 📊 How the Recommendations Work

### Price-Based Signal (Tab 1)
```python
if price_change > 5%:
    → BUY (green)
elif price_change < -5%:
    → SELL (red)
else:
    → HOLD (orange)
```

### Portfolio Recommendations (Tab 2)
Uses **Capital Asset Pricing Model (CAPM)**:
- Calculates expected returns vs market (TUNINDEX)
- Computes Beta (volatility relative to market)
- Considers risk profile constraints
- Detects anomalies for each stock
- Generates BUY/SELL/HOLD based on:
  - Expected return vs risk
  - Current vs optimal allocation
  - Risk profile limits
  - Anomaly presence

---

## 🔧 Backend Requirements

Make sure these endpoints are working:
- `/api/v1/trading/predictions` — Price predictions
- `/api/v1/trading/sentiment` — Sentiment scores
- `/api/v1/trading/anomalies` — Anomaly detection
- `/api/v1/trading/predictions/volume` — Volume forecasts
- `/api/v1/trading/predictions/liquidity` — Liquidity forecasts
- `/api/v1/ai/portfolio/recommendations/detailed` — Portfolio recommendations

---

## 💡 Tips

1. **Start with Smart Trading page** — it has everything you need
2. **Enable AI Explanations** in Tab 2 for human-readable insights
   - You'll need a Groq API key (free at groq.com)
   - Or use OpenRouter/OpenAI
3. **Check anomalies** — red flags indicate unusual activity
4. **Compare multiple stocks** in Tab 2 for portfolio optimization
5. **Use Full Report** (Tab 3) for deep dives on specific stocks

---

## 🐛 Troubleshooting

### "Backend not available" error
- Make sure FastAPI is running: `uvicorn app.main:app --port 8000`
- Check logs: Terminal should show `INFO:     Uvicorn running on http://127.0.0.1:8000`

### No predictions/recommendations
- Database might be empty — run ETL pipeline first
- Check if stock symbol exists in `stock_prices` table
- Run training: `python -m prediction train`

### Slow performance
- First request is slow (model loading)
- Subsequent requests use cache (< 50ms)
- Enable Redis for better caching

---

## 📚 Other Pages

- **🏠 Dashboard** — Quick overview and status
- **📊 Price Prediction** — Detailed price forecasting
- **📈 Volume & Liquidity** — Transaction volume and liquidity analysis
- **💬 Sentiment Analysis** — NLP sentiment from news articles
- **🚨 Anomaly Detection** — Market anomaly alerts
- **💼 AI Portfolio** — Portfolio optimization and backtesting
- **🎯 Recommendations** — Alternative recommendation interface
- **⚙️ Settings & Status** — System health and API explorer

---

Built with ❤️ for BVMT (Bourse des Valeurs Mobilières de Tunis)
