"""
Quick Start Guide for AI Module.

Follow these steps to get started with the AI Decision Agent.
"""

# ============================================================
# QUICK START - AI MODULE
# ============================================================

## 1. Installation

# Install dependencies
pip install groq
# Or update from requirements.txt
pip install -r requirements.txt


## 2. Configuration

# Add to your .env file:
"""
# Groq AI API (get key from https://console.groq.com)
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1024
GROQ_TEMPERATURE=0.7

# Portfolio Settings (optional, defaults provided)
DEFAULT_INITIAL_CAPITAL=10000.0
DEFAULT_RISK_PROFILE=moderate
"""


## 3. Run the Application

# Start FastAPI server
cd fixtrade
uvicorn app.main:app --reload

# API will be available at:
# http://localhost:8000
# Documentation: http://localhost:8000/docs


## 4. Test Examples

# Run example scenarios
python -m app.ai.examples


## 5. API Endpoints

### Get AI Module Status
GET http://localhost:8000/api/v1/ai/status

### Profile Assessment
POST http://localhost:8000/api/v1/ai/profile/questionnaire
Body:
{
  "age": 28,
  "investment_horizon": 5,
  "income_stability": "high",
  "investment_experience": "beginner",
  "loss_tolerance": 3,
  "financial_goals": "growth"
}

### Create Portfolio
POST http://localhost:8000/api/v1/ai/portfolio/create
Body:
{
  "risk_profile": "moderate",
  "initial_capital": 10000.0
}

### Get Portfolio Snapshot
GET http://localhost:8000/api/v1/ai/portfolio/default/snapshot

### Get Performance Metrics
GET http://localhost:8000/api/v1/ai/portfolio/default/performance

### Execute Trade
POST http://localhost:8000/api/v1/ai/portfolio/default/trade
Body:
{
  "symbol": "AMEN",
  "action": "buy",
  "quantity": 100,
  "price": 12.50,
  "generate_explanation": true
}

### Update Market Prices
POST http://localhost:8000/api/v1/ai/portfolio/default/prices/update
Body:
{
  "AMEN": 15.00,
  "ATTIJARI": 26.50,
  "BNA": 5.50
}

### Check Stop-Loss
POST http://localhost:8000/api/v1/ai/portfolio/default/stop-loss/check
Body:
{
  "AMEN": 11.00,
  "ATTIJARI": 23.00
}


## 6. Programmatic Usage

from app.ai import DecisionAgent, RiskProfile

# Create agent
agent = DecisionAgent(
    risk_profile=RiskProfile.MODERATE,
    initial_capital=10000.0
)

# Execute trade
result = await agent.execute_trade(
    session=db_session,
    symbol="AMEN",
    action="buy",
    quantity=100,
    price=12.50
)

# Get metrics
metrics = agent.get_performance_metrics()
print(f"ROI: {metrics['roi']:.2f}%")


## 7. Troubleshooting

### Groq API Not Working?
- Check GROQ_API_KEY is set correctly
- Verify API key permissions at https://console.groq.com
- Module will fall back to rule-based explanations

### Database Connection?
- Ensure PostgreSQL is running
- Check DATABASE_URL in .env
- Some features require DB integration (work in progress)

### Import Errors?
- Ensure all dependencies installed: pip install -r requirements.txt
- Check Python version >= 3.10


## 8. Next Steps

- Review README.md for detailed documentation
- Check examples.py for usage scenarios
- Explore API docs at /docs endpoint
- Integrate with frontend (see front_fixtrade/PREDICTION_INTEGRATION.md)


## 9. Support

- Architecture: Clean Architecture / Domain-Driven Design
- API Documentation: Auto-generated at /docs
- Logs: Check console output for detailed logging
- Issues: File in project repository
