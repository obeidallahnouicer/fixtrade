# AI Module Architecture & Data Flow

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Frontend/API Consumer)                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTP Requests
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTERFACES LAYER (FastAPI)                          │
│                           app/ai/router.py                                  │
│                                                                             │
│  Endpoints:                                               Status:           │
│  • POST /ai/profile/questionnaire                         ✅ Works         │
│  • POST /ai/portfolio/create                              ✅ Works         │
│  • GET  /ai/recommendations                               🔧 Needs session │
│  • GET  /ai/recommendations/{symbol}/explain              🔧 Needs session │
│  • POST /ai/portfolio/{id}/trade                          🔧 Needs session │
│  • GET  /ai/portfolio/{id}/snapshot                       ✅ Works         │
│  • GET  /ai/portfolio/{id}/performance                    ✅ Works         │
│  • GET  /ai/portfolio/{id}/performance/chart              ❌ Missing       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                  │
│                         app/ai/agent.py                                     │
│                       (Decision Agent)                                      │
│                                                                             │
│  Responsibilities:                                        Status:           │
│  • Orchestrate all AI components                          ✅ Good          │
│  • Manage portfolio lifecycle                             ✅ Good          │
│  • Execute trades with risk checks                        ✅ Good          │
│  • Generate recommendations                               ✅ Good          │
│  • Calculate performance metrics                          ✅ Good          │
│  • Provide explanations                                   ✅ Good          │
│  • Persist to database                                    ❌ Missing       │
└─────┬───────────────┬───────────────┬────────────────┬────────────────────┘
      │               │               │                │
      ▼               ▼               ▼                ▼
┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐
│ Portfolio │  │Recommendation│  │  Metrics │  │Explainability│
│  Manager  │  │   Engine     │  │Calculator│  │  Generator   │
│  ✅ DONE  │  │   ✅ DONE    │  │ ✅ DONE  │  │   ✅ DONE    │
└─────┬─────┘  └──────┬───────┘  └──────────┘  └──────┬───────┘
      │                │                               │
      │                ▼                               ▼
      │         ┌──────────────┐              ┌────────────────┐
      │         │     Rules    │              │   Groq API     │
      │         │    Engine    │              │  (External)    │
      │         │   ✅ DONE    │              │   ✅ Works     │
      │         └──────┬───────┘              └────────────────┘
      │                │
      │                ▼
      │         ┌──────────────┐
      │         │     Data     │
      │         │  Aggregator  │
      │         │ 🔧 PARTIAL   │
      │         └──────┬───────┘
      │                │
      ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                │
│                   (Database & External Services)                            │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PostgreSQL  │  │  Prediction  │  │   Sentiment  │  │   Anomaly    │  │
│  │   Database   │  │   Service    │  │   Analysis   │  │  Detection   │  │
│  │              │  │              │  │              │  │              │  │
│  │ Tables:      │  │ Status:      │  │ Status:      │  │ Status:      │  │
│  │ • stocks     │  │ 🔧 Not wired │  │ ✅ Working   │  │ ✅ Working   │  │
│  │ • sentiment  │  │              │  │              │  │              │  │
│  │ • anomalies  │  │              │  │              │  │              │  │
│  │ • portfolios │  │              │  │              │  │              │  │
│  │   ❌ Missing │  │              │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Get Recommendations

```
┌────────┐
│ Client │
└───┬────┘
    │ GET /ai/recommendations?portfolio_id=123&top_n=5
    ▼
┌───────────────────┐
│  FastAPI Router   │
│                   │
│ 1. Parse request  │ ✅ Works
│ 2. Get DB session │ 🔧 NEEDS: Depends(get_session)
│ 3. Load agent     │ 🔧 NEEDS: Load from DB
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Decision Agent    │
│                   │
│ get_daily_recs()  │
└───────┬───────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Recommendation Engine                                 │
│                                                       │
│ 1. Get candidate symbols                             │ ✅ Logic ready
│    → Query top movers, user watchlist                │
│                                                       │
│ 2. For each symbol, get signals                      │
│    └─→ Data Aggregator                               │
│        ├─→ Current price (DB)           🔧 NEEDS    │
│        ├─→ Prediction (Service)         🔧 NEEDS    │
│        ├─→ Sentiment (DB)               ✅ Works    │
│        ├─→ Anomaly (DB)                 ✅ Works    │
│        └─→ Liquidity (Service)          🔧 NEEDS    │
│                                                       │
│ 3. Evaluate with rules                               │
│    └─→ Rule Engine                                   │
│        • Check prediction confidence                 │ ✅ Logic ready
│        • Check sentiment score                       │ ✅ Logic ready
│        • Check anomaly severity                      │ ✅ Logic ready
│        • Apply risk profile filters                  │ ✅ Logic ready
│        • Generate signal (BUY/SELL/HOLD)             │ ✅ Logic ready
│                                                       │
│ 4. Generate explanations                             │
│    └─→ Explainability Generator                      │
│        • Build context                               │ ✅ Works
│        • Call Groq API                               │ ✅ Works
│        • Return natural language                     │ ✅ Works
│                                                       │
│ 5. Rank and filter                                   │
│    • Sort by score                                   │ ✅ Logic ready
│    • Filter by risk profile                          │ ✅ Logic ready
│    • Return top N                                    │ ✅ Logic ready
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ Response          │
│                   │
│ [                 │
│   {               │
│     "symbol": "AMEN",                                │
│     "signal": "BUY",                                 │
│     "strength": "HIGH",                              │
│     "explanation": "Strong buy signal...",           │
│     "confidence": 0.85,                              │
│     "predicted_return": 3.5                          │
│   },                                                 │
│   ...                                                │
│ ]                                                    │
└───────────────────┘
```

---

## Data Flow: Execute Trade

```
┌────────┐
│ Client │
└───┬────┘
    │ POST /ai/portfolio/123/trade
    │ {
    │   "symbol": "AMEN",
    │   "action": "buy",
    │   "quantity": 10,
    │   "price": 12.50
    │ }
    ▼
┌───────────────────────────────────────────────────────┐
│ FastAPI Router                                        │
│                                                       │
│ 1. Validate input (Pydantic)         ✅ Works        │
│ 2. Get DB session                    🔧 NEEDS        │
│ 3. Load agent from DB                🔧 NEEDS        │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Decision Agent                                        │
│                                                       │
│ execute_trade()                                       │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Portfolio Manager                                     │
│                                                       │
│ 1. Check risk limits                 ✅ Works        │
│    • Sufficient cash?                                │
│    • Position size OK?                               │
│    • Equity allocation OK?                           │
│                                                       │
│ 2. Execute trade                     ✅ Works        │
│    • Update positions                                │
│    • Update cash balance                             │
│    • Record trade history                            │
│                                                       │
│ 3. Check stop-losses                 ✅ Works        │
│    • Any positions below threshold?                  │
│    • Auto-sell if triggered                          │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Explainability Generator                              │
│                                                       │
│ Generate explanation for trade      ✅ Works         │
│ • Why this trade makes sense                         │
│ • Impact on portfolio                                │
│ • Risk considerations                                │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Database Persistence                                  │
│                                                       │
│ 1. Save portfolio state              ❌ MISSING      │
│ 2. Save position                     ❌ MISSING      │
│ 3. Save trade record                 ❌ MISSING      │
│ 4. Save snapshot                     ❌ MISSING      │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ Response          │
│                   │
│ {                                                     │
│   "success": true,                                    │
│   "message": "Achat réussi: 10 AMEN @ 12.50 TND",   │
│   "portfolio_value": 9875.0,                         │
│   "explanation": "This purchase aligns with..."      │
│ }                                                     │
└───────────────────┘
```

---

## Data Flow: Calculate Performance

```
┌────────┐
│ Client │
└───┬────┘
    │ GET /ai/portfolio/123/performance
    ▼
┌───────────────────────────────────────────────────────┐
│ FastAPI Router                                        │
│                                                       │
│ 1. Parse portfolio_id                ✅ Works        │
│ 2. Load agent from DB                🔧 NEEDS        │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Decision Agent                                        │
│                                                       │
│ get_performance_metrics()                             │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Portfolio Manager                                     │
│                                                       │
│ calculate_metrics()                                   │
│                                                       │
│ Extracts:                            ✅ Works        │
│ • Value history                                       │
│ • Trade history                                       │
│ • Returns time series                                 │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Metrics Calculator                                    │
│                                                       │
│ calculate_all_metrics()              ✅ Works        │
│                                                       │
│ Calculates:                                           │
│ • ROI                                                 │
│ • Sharpe Ratio                                        │
│ • Maximum Drawdown                                    │
│ • Volatility                                          │
│ • Win Rate                                            │
│ • Profit Factor                                       │
│ • Annualized Return                                   │
└───────┬───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ Response          │
│                   │
│ {                                                     │
│   "total_value": 11250.0,                            │
│   "total_return": 12.5,                              │
│   "roi": 12.5,                                       │
│   "sharpe_ratio": 1.85,                              │
│   "max_drawdown": -3.2,                              │
│   "volatility": 15.3,                                │
│   "win_rate": 75.0,                                  │
│   "total_trades": 8,                                 │
│   "winning_trades": 6,                               │
│   "losing_trades": 2                                 │
│ }                                                     │
└───────────────────┘
```

---

## Critical Integration Points

### 🔴 Blocker 1: Database Session Injection

**Current State**:
```python
@router.get("/recommendations")
async def get_daily_recommendations(...):
    # TODO: Get database session
    session = None  # ❌ Broken
```

**Required State**:
```python
from app.core.db import get_session

@router.get("/recommendations")
async def get_daily_recommendations(
    session: AsyncSession = Depends(get_session)  # ✅ Works
):
    ...
```

---

### 🔴 Blocker 2: Portfolio Persistence

**Current State**:
```python
# In-memory only
_agents: Dict[str, DecisionAgent] = {}  # ❌ Lost on restart
```

**Required State**:
```python
# Load from database
portfolio = await portfolio_repo.load_portfolio(session, portfolio_id)
agent = DecisionAgent(...)
agent.portfolio = portfolio  # ✅ Persists
```

**Missing Tables**:
- `portfolios`
- `portfolio_positions`
- `portfolio_trades`
- `portfolio_snapshots`

---

### 🔴 Blocker 3: Data Aggregator Integration

**Current State**:
```python
async def _get_current_price(self, symbol, session):
    # TODO: Implement
    return None  # ❌ Returns nothing
```

**Required State**:
```python
async def _get_current_price(self, symbol, session):
    query = select(StockPrice.close).where(...)
    result = await session.execute(query)
    return result.scalar_one_or_none()  # ✅ Returns real price
```

---

## Component Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│ Legend:                                                     │
│ ✅ Ready   🔧 Needs Work   ❌ Missing   → Depends On        │
└─────────────────────────────────────────────────────────────┘

DecisionAgent (✅ Ready)
├─→ PortfolioManager (✅ Ready)
│   ├─→ UserProfileManager (✅ Ready)
│   ├─→ MetricsCalculator (✅ Ready)
│   └─→ Database (❌ Missing)
│
├─→ RecommendationEngine (✅ Ready)
│   ├─→ DataAggregator (🔧 Needs Work)
│   │   ├─→ Database (🔧 Needs Queries)
│   │   ├─→ PredictionService (❌ Not Wired)
│   │   └─→ SentimentAnalyzer (✅ Ready)
│   │
│   ├─→ RuleBasedEngine (✅ Ready)
│   └─→ ExplanationGenerator (✅ Ready)
│       └─→ Groq API (✅ Ready)
│
└─→ MetricsCalculator (✅ Ready)
```

---

## Database Schema (Required)

```sql
-- Missing tables for persistence

CREATE TABLE portfolios (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    risk_profile VARCHAR(50) CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    initial_capital FLOAT NOT NULL,
    cash_balance FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE portfolio_positions (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    purchase_price FLOAT NOT NULL,
    purchased_at DATE NOT NULL,
    current_price FLOAT
);

CREATE TABLE portfolio_trades (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) CHECK (action IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL,
    price FLOAT NOT NULL,
    executed_at TIMESTAMP NOT NULL,
    profit_loss FLOAT
);

CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    total_value FLOAT NOT NULL,
    cash_balance FLOAT NOT NULL,
    equity_value FLOAT NOT NULL
);
```

---

## External Service Integration

### Prediction Service

**Current State**: ❌ Not integrated

**Required Integration**:
```python
# In router.py
async def get_prediction_service():
    from prediction.inference import PredictionService
    return PredictionService()

@router.get("/recommendations")
async def get_daily_recommendations(
    prediction_svc = Depends(get_prediction_service)
):
    recommendations = await agent.get_daily_recommendations(
        prediction_service=prediction_svc  # ✅ Now wired
    )
```

---

## Error Handling Flow

```
┌────────┐
│Request │
└───┬────┘
    │
    ▼
┌──────────────┐
│ Validation   │ Pydantic models
│ (FastAPI)    │ ✅ Works
└───┬──────────┘
    │ Valid?
    ├─No──→ 422 Unprocessable Entity
    │
    ▼ Yes
┌──────────────┐
│ Business     │
│ Logic        │
│ (Agent)      │
└───┬──────────┘
    │ Success?
    ├─No──→ Custom exception
    │       └─→ HTTPException with detail
    │           ✅ Safe error messages
    │
    ▼ Yes
┌──────────────┐
│ Database     │
│ Operation    │ ❌ Missing
└───┬──────────┘
    │ Success?
    ├─No──→ SQLAlchemy error
    │       └─→ Rollback + log
    │           └─→ 500 Internal Server Error
    │
    ▼ Yes
┌──────────────┐
│ Response     │
│ (Success)    │
└──────────────┘
```

---

## Performance Bottlenecks

### Identified Issues

1. **N+1 Database Queries**
   ```python
   # BAD: Multiple queries
   for symbol in symbols:
       price = await get_price(symbol)      # Query 1
       sentiment = await get_sentiment(symbol)  # Query 2
       anomaly = await get_anomaly(symbol)      # Query 3
   
   # GOOD: Single JOIN query
   results = await get_all_signals(symbols)  # 1 query total
   ```

2. **No Caching**
   ```python
   # BAD: Fetch every time
   signals = await aggregator.get_signals(symbol, session)
   
   # GOOD: Cache for 5 minutes
   if cached := cache.get(symbol):
       return cached
   signals = await aggregator.get_signals(symbol, session)
   cache.set(symbol, signals, ttl=300)
   ```

3. **Synchronous Groq API**
   ```python
   # BAD: Blocks thread
   explanation = client.chat.completions.create(...)
   
   # GOOD: Async with timeout
   async with httpx.AsyncClient() as client:
       response = await client.post(..., timeout=5.0)
   ```

---

## Monitoring Points

### Metrics to Track

1. **Request Metrics**
   - Requests per minute (by endpoint)
   - Response time (P50, P95, P99)
   - Error rate

2. **Business Metrics**
   - Recommendations generated/day
   - Trades executed/day
   - Average portfolio ROI
   - Active portfolios

3. **System Metrics**
   - Database connection pool usage
   - Groq API latency
   - Memory usage per agent
   - Cache hit rate

### Logging Strategy

```python
# Request level
logger.info(f"GET /recommendations portfolio_id={id} top_n={n}")

# Business level
logger.info(f"Generated {len(recs)} recommendations for {portfolio_id}")

# Error level
logger.error(f"Failed to fetch signals for {symbol}: {error}")

# Performance level
logger.debug(f"Cache hit for {symbol}")
```

---

## Security Boundaries

```
┌────────────────────────────────────────────────────────┐
│ PUBLIC INTERNET                                        │
│ • No authentication (out of scope)                     │
│ • Rate limiting (❌ MISSING)                           │
│ • Input validation (✅ DONE)                           │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ FASTAPI LAYER                                          │
│ • Pydantic validation (✅ DONE)                        │
│ • Request size limits (⚠️ NEEDS)                       │
│ • Safe error messages (✅ DONE)                        │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ APPLICATION LAYER                                      │
│ • Risk limit enforcement (✅ DONE)                     │
│ • Portfolio boundaries (✅ DONE)                       │
│ • No code execution (✅ SAFE)                          │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ DATABASE LAYER                                         │
│ • SQLAlchemy ORM (✅ SAFE)                             │
│ • Parameterized queries (✅ DONE)                      │
│ • No raw SQL (✅ DONE)                                 │
└────────────────────────────────────────────────────────┘
```

---

## Testing Coverage Map

```
Component              Unit Tests    Integration Tests    Status
─────────────────────────────────────────────────────────────────
PortfolioManager       ❌ Missing    ❌ Missing          Priority: HIGH
MetricsCalculator      ❌ Missing    N/A                 Priority: HIGH
RuleBasedEngine        ❌ Missing    N/A                 Priority: MEDIUM
DecisionAgent          ❌ Missing    ❌ Missing          Priority: HIGH
RecommendationEngine   ❌ Missing    ❌ Missing          Priority: MEDIUM
DataAggregator         ❌ Missing    ❌ Missing          Priority: MEDIUM
ExplanationGenerator   ❌ Missing    N/A                 Priority: LOW
UserProfileManager     ❌ Missing    N/A                 Priority: LOW

Full Trading Flow      N/A           ❌ Missing          Priority: HIGH
```

---

**Document Version**: 1.0  
**Created**: February 8, 2026  
**Purpose**: Visual architecture reference  
**Audience**: Development team
