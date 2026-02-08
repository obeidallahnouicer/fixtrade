# 🚀 Quick Start Guide - Portfolio Optimization

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Optimize Portfolio (Max Sharpe)
```python
from app.ai.optimization import PortfolioOptimizer
from app.ai.profile import RiskProfile
import numpy as np

# Your historical returns (250 days, 5 assets)
returns = np.random.randn(250, 5) * 0.02 + 0.001
symbols = ["BNA", "STB", "BIAT", "SOTUMAG", "SOTRAPIL"]

optimizer = PortfolioOptimizer(
    returns=returns,
    symbols=symbols,
    risk_profile=RiskProfile.MODERATE
)

result = optimizer.maximum_sharpe_portfolio()

print(result.weights)
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### 2. Generate Trading Signals
```python
from app.ai.decision_engine import DecisionEngine

engine = DecisionEngine(risk_profile=RiskProfile.MODERATE)

recommendations = engine.generate_recommendations(
    symbols=symbols,
    returns=returns,
    market_returns=market_returns,  # TUNINDEX
    current_weights={},
    anomalies={"BNA": False, "STB": False, ...}
)

for rec in recommendations:
    print(f"{rec['symbol']}: {rec['action']} ({rec['confidence']:.0f}%)")
    print(f"→ {rec['explanation']}\n")
```

### 3. Run Backtest
```python
from app.ai.simulator import PortfolioSimulator
from datetime import date

sim = PortfolioSimulator(initial_capital=10000.0)

# Execute trades
sim.buy("BNA", 100, 45.0, date.today())
sim.buy("STB", 200, 12.5, date.today())

# Record state
prices = {"BNA": 45.0, "STB": 12.5}
sim.record_state(date.today(), prices)

# Get metrics
metrics = sim.calculate_metrics()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Max DD: {metrics.max_drawdown:.1f}%")
```

### 4. Generate Explanations
```python
from app.ai.llm_explainer import LLMExplainer, ExplanationContext

explainer = LLMExplainer(use_llm=False)  # Template mode

ctx = ExplanationContext(
    symbol="BNA",
    action="BUY",
    confidence=85.0,
    current_weight=0.0,
    target_weight=15.0,
    expected_return=12.5,
    beta=0.85,
    risk_contribution=8.2,
    anomaly_detected=False
)

print(explainer.explain_decision(ctx))
```

## API Usage

### Optimize Portfolio
```bash
curl -X POST http://localhost:8000/ai/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BNA", "STB", "BIAT"],
    "risk_profile": "moderate",
    "optimization_method": "max_sharpe"
  }'
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/ai/portfolio/recommendations/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BNA", "STB", "BIAT", "SOTUMAG"],
    "risk_profile": "moderate",
    "current_weights": {"BNA": 0.3, "STB": 0.2},
    "use_llm_explanation": false
  }'
```

### Get Efficient Frontier
```bash
curl -X POST http://localhost:8000/ai/portfolio/efficient-frontier \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BNA", "STB", "BIAT", "SOTUMAG"],
    "num_points": 50
  }'
```

## Configuration

### Risk Profiles
```python
from app.ai.profile import RiskProfile

# Three profiles available:
RiskProfile.CONSERVATIVE  # Low risk, high diversification
RiskProfile.MODERATE      # Balanced risk-return
RiskProfile.AGGRESSIVE    # High risk, high return potential
```

### Optimization Settings
```python
# In app/ai/config.py
RISK_FREE_RATE = 0.05           # 5% annual (TND bonds)
MARKET_RISK_PREMIUM = 0.08      # 8%
TRADING_DAYS_PER_YEAR = 250     # BVMT calendar
```

## Common Patterns

### Full Pipeline
```python
async def get_recommendations(symbols: List[str]):
    # 1. Fetch data
    returns = await fetch_historical_returns(db, symbols)
    market_returns = await fetch_market_returns(db)
    anomalies = await fetch_anomaly_status(db, symbols)
    
    # 2. Generate signals
    engine = DecisionEngine(risk_profile=RiskProfile.MODERATE)
    recommendations = engine.generate_recommendations(
        symbols=symbols,
        returns=returns,
        market_returns=market_returns,
        current_weights={},
        anomalies=anomalies
    )
    
    return recommendations
```

### Database Integration
```python
async def fetch_historical_returns(db, symbols, days=250):
    query = """
    SELECT symbol, date, close
    FROM stock_prices
    WHERE symbol = ANY(:symbols)
      AND date >= CURRENT_DATE - :days
    ORDER BY date
    """
    result = await db.execute(query, {"symbols": symbols, "days": days})
    # Process into returns matrix...
    return returns
```

## Troubleshooting

### Optimization Fails
- Check returns matrix shape: (days, num_assets)
- Ensure no NaN/Inf values
- Verify at least 100 days of history
- Check covariance matrix is positive definite

### No BUY Signals
- Verify expected returns > risk-free rate
- Check beta values (should be 0.5-2.0)
- Confirm risk profile constraints aren't too tight
- Review anomaly detection results

### Poor Sharpe Ratio
- Check risk-free rate setting
- Verify market returns are reasonable
- Review asset correlation (high correlation = low diversification)
- Consider longer historical period

## File Locations
```
app/ai/
├── optimization.py          # Portfolio optimization
├── decision_engine.py       # BUY/SELL/HOLD signals
├── simulator.py             # Backtesting
├── llm_explainer.py         # Explanations
└── router_extended.py       # API endpoints

docs/
├── PORTFOLIO_OPTIMIZATION_IMPLEMENTATION.md  # Full docs
├── INTEGRATION_EXAMPLE.py                    # Code examples
└── DELIVERABLES.md                           # Project summary
```

## Support
See full documentation: `docs/PORTFOLIO_OPTIMIZATION_IMPLEMENTATION.md`
