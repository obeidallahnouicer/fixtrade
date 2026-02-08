# AI Module - Action Plan

## Quick Status Overview

```
┌─────────────────────────────────────────────────────────────┐
│  AI Module Implementation Status: 75% Complete              │
│  Production Readiness: NOT READY (5 days to MVP)            │
│  Code Quality: 8.5/10                                       │
│  Security Compliance: ✅ GOOD                                │
└─────────────────────────────────────────────────────────────┘
```

## Requirement Compliance Matrix

| Requirement | Status | Implementation | Gap |
|------------|--------|----------------|-----|
| **1. Profil Utilisateur** | ✅ | `profile.py` + questionnaire | None |
| **2. Agrégation Intelligente** | 🔧 | `aggregator.py` partial | Missing prediction service wiring |
| **3. Simulation Portefeuille** | ✅ | `portfolio.py` complete | Need DB persistence |
| **4. Métriques (ROI, Sharpe, etc.)** | ✅ | `metrics.py` complete | None |
| **5. Explainability** | ✅ | `explainability.py` + Groq | None |
| **6. Recommandations** | ✅ | `recommendations.py` + `rules.py` | None |
| **7. Vue Portefeuille (API)** | 🔧 | `router.py` endpoints exist | Missing session injection |
| **8. Bouton "Expliquer"** | ✅ | `/recommendations/{symbol}/explain` | None |
| **9. Graphique Performance** | ❌ | Data collected, no endpoint | Need visualization API |

**Legend**: ✅ Complete | 🔧 Partial | ❌ Missing

---

## Critical Blockers (MUST FIX)

### 🔴 Blocker 1: No Database Persistence
**Impact**: Portfolio state lost on restart  
**Files Affected**: `agent.py`, `portfolio.py`, `router.py`  
**Effort**: 2 days  

**Solution**:
1. Create DB models: `Portfolio`, `PortfolioPosition`, `PortfolioTrade`
2. Add migration: `db/002_portfolio_tables.sql`
3. Implement `portfolio.load_from_db()` and `portfolio.save_to_db()`
4. Update router to use DB-backed agents

**Code Snippet**:
```python
# New file: app/infrastructure/trading/portfolio_repository.py
class PortfolioRepository:
    async def save_portfolio(self, session, portfolio: PortfolioManager):
        """Persist portfolio to database."""
        pass
    
    async def load_portfolio(self, session, portfolio_id: UUID) -> PortfolioManager:
        """Load portfolio from database."""
        pass
```

---

### 🔴 Blocker 2: API Endpoints Non-Functional
**Impact**: Cannot test AI module end-to-end  
**Files Affected**: `router.py`, `aggregator.py`  
**Effort**: 1 day  

**Solution**:
```python
# BEFORE (broken):
@router.get("/recommendations")
async def get_daily_recommendations(...):
    # TODO: Get database session
    # TODO: Get prediction service
    return []  # Returns nothing

# AFTER (working):
@router.get("/recommendations")
async def get_daily_recommendations(
    session: AsyncSession = Depends(get_session),  # ✅ INJECT
    prediction_svc = Depends(get_prediction_service)  # ✅ INJECT
):
    agent = get_or_create_agent(portfolio_id)
    recommendations = await agent.get_daily_recommendations(
        session=session,
        prediction_service=prediction_svc  # ✅ WORKS NOW
    )
    return [rec.to_dict() for rec in recommendations]
```

---

### 🔴 Blocker 3: Data Aggregator Incomplete
**Impact**: Recommendations don't use real data  
**Files Affected**: `aggregator.py`  
**Effort**: 1 day  

**Functions to Complete**:
- `_get_current_price()` - Query `stock_prices` table
- `_get_sentiment()` - Query `sentiment_scores` table
- `_get_anomaly()` - Query `anomalies` table
- `_get_prediction()` - Call prediction service
- `_get_volume_prediction()` - Call prediction service
- `_get_liquidity_prediction()` - Call prediction service

---

## High Priority Enhancements (SHOULD HAVE)

### 🟡 Enhancement 1: Performance Visualization
**Business Value**: HIGH - Users need to see portfolio growth  
**Effort**: 0.5 days  

**New Endpoint**:
```python
@router.get("/portfolio/{portfolio_id}/performance/chart")
async def get_performance_chart(portfolio_id: str):
    """
    Return time-series data for portfolio performance graph.
    
    Response:
    {
        "dates": ["2026-01-01", "2026-01-02", ...],
        "values": [10000, 10050, 10120, ...],
        "roi": [0.0, 0.5, 1.2, ...],
        "benchmark": [...]  # TUNINDEX for comparison
    }
    """
    pass
```

---

### 🟡 Enhancement 2: Portfolio Optimization
**Business Value**: MEDIUM - Helps users rebalance  
**Effort**: 1 day  

**New Method**:
```python
# Add to portfolio.py
def suggest_rebalancing(self) -> List[Dict[str, Any]]:
    """
    Suggest trades to rebalance portfolio to target allocation.
    
    Returns:
        [
            {"action": "SELL", "symbol": "AMEN", "quantity": 5, "reason": "..."},
            {"action": "BUY", "symbol": "BNA", "quantity": 10, "reason": "..."}
        ]
    """
    pass
```

---

### 🟡 Enhancement 3: Rate Limiting
**Business Value**: HIGH - Prevent abuse  
**Effort**: 0.5 days  

**Implementation**:
```python
# Add to router.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/recommendations")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def get_daily_recommendations(...):
    pass
```

---

## 5-Day Implementation Plan

### Day 1: Database Persistence
**Goal**: Portfolio state persists across restarts

**Tasks**:
- [ ] Create `Portfolio`, `PortfolioPosition`, `PortfolioTrade` models
- [ ] Write migration script: `db/002_portfolio_tables.sql`
- [ ] Implement repository pattern: `portfolio_repository.py`
- [ ] Add `load_from_db()` and `save_to_db()` to `PortfolioManager`
- [ ] Test with manual insert/select

**Acceptance**:
- ✅ Can create portfolio and retrieve it after restart
- ✅ Positions and trades persist correctly

---

### Day 2: Complete Data Aggregator
**Goal**: All signals use real database data

**Tasks**:
- [ ] Implement `_get_current_price()` with real query
- [ ] Implement `_get_sentiment()` with real query
- [ ] Implement `_get_anomaly()` with real query
- [ ] Add error handling for missing data
- [ ] Add caching layer (5-minute TTL)
- [ ] Test with real stock symbols

**Acceptance**:
- ✅ `get_signals()` returns real data for all fields
- ✅ No "TODO" comments remaining

---

### Day 3: Wire API Endpoints
**Goal**: All endpoints functional end-to-end

**Tasks**:
- [ ] Add `get_session` dependency injection to all endpoints
- [ ] Create `get_prediction_service` dependency
- [ ] Update `get_or_create_agent()` to load from DB
- [ ] Remove all placeholder returns
- [ ] Add proper error handling
- [ ] Test each endpoint with Postman/curl

**Acceptance**:
- ✅ `/recommendations` returns real recommendations
- ✅ `/portfolio/{id}/snapshot` shows actual data
- ✅ `/portfolio/{id}/trade` executes and persists

---

### Day 4: Enhancements
**Goal**: Add missing MVP features

**Tasks**:
- [ ] Add `/portfolio/{id}/performance/chart` endpoint
- [ ] Implement `suggest_rebalancing()` method
- [ ] Add rate limiting to all endpoints
- [ ] Add request timeout to Groq API calls
- [ ] Improve error messages

**Acceptance**:
- ✅ Performance chart data available
- ✅ Rate limiting prevents spam
- ✅ All operations complete in < 5s

---

### Day 5: Testing & Documentation
**Goal**: Production-ready module

**Tasks**:
- [ ] Write unit tests for `portfolio.py` (target: 90% coverage)
- [ ] Write unit tests for `metrics.py` (target: 95% coverage)
- [ ] Write integration test for full trading flow
- [ ] Update API documentation (OpenAPI descriptions)
- [ ] Create user guide for risk profiles
- [ ] Code review with team

**Acceptance**:
- ✅ Test coverage > 80%
- ✅ All critical paths tested
- ✅ Documentation complete

---

## Testing Checklist

### Manual Testing Scenarios

#### Scenario 1: New User Onboarding
```bash
# 1. Determine risk profile
curl -X POST http://localhost:8000/api/v1/ai/profile/questionnaire \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "investment_horizon": 5,
    "income_stability": "high",
    "investment_experience": "beginner",
    "loss_tolerance": 3,
    "financial_goals": "growth"
  }'

# Expected: {"recommended_profile": "moderate", ...}

# 2. Create portfolio
curl -X POST http://localhost:8000/api/v1/ai/portfolio/create \
  -H "Content-Type: application/json" \
  -d '{"risk_profile": "moderate", "initial_capital": 10000}'

# Expected: {"portfolio_id": "...", "initial_capital": 10000, ...}
```

#### Scenario 2: Get Recommendations and Execute Trade
```bash
# 1. Get recommendations
curl http://localhost:8000/api/v1/ai/recommendations?portfolio_id=XXX&top_n=5

# Expected: List of 5 recommendations with explanations

# 2. Execute buy trade
curl -X POST http://localhost:8000/api/v1/ai/portfolio/XXX/trade \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AMEN",
    "action": "buy",
    "quantity": 10,
    "price": 12.50
  }'

# Expected: {"success": true, "message": "Achat réussi", ...}

# 3. Check portfolio
curl http://localhost:8000/api/v1/ai/portfolio/XXX/snapshot

# Expected: Cash reduced, position added
```

#### Scenario 3: Stop-Loss Trigger
```bash
# 1. Buy stock
curl -X POST .../trade -d '{"symbol": "AMEN", "action": "buy", "quantity": 10, "price": 100}'

# 2. Update price (simulate loss)
curl -X POST .../prices/update -d '{"AMEN": 90}'

# 3. Check stop-loss
curl -X POST .../stop-loss/check -d '{"AMEN": 90}'

# Expected: Stop-loss triggered, position sold
```

---

## Performance Targets

| Endpoint | Target Response Time | Max Load |
|----------|---------------------|----------|
| `GET /recommendations` | < 2s | 10 req/min per user |
| `POST /trade` | < 500ms | 20 req/min per user |
| `GET /portfolio/snapshot` | < 100ms | 60 req/min per user |
| `GET /performance` | < 1s | 30 req/min per user |

**Optimization Strategies**:
- Cache signals for 5 minutes
- Batch database queries
- Use async Groq API calls
- Add database indexes on `symbol`, `date`, `portfolio_id`

---

## Success Metrics

### Technical Metrics
- ✅ Test coverage > 80%
- ✅ All endpoints < 3s response time
- ✅ Zero SQL injection vulnerabilities
- ✅ All inputs validated
- ✅ No memory leaks (< 500MB steady-state)

### Business Metrics
- ✅ Can create and manage portfolio
- ✅ Can get daily recommendations
- ✅ Can execute trades with explanations
- ✅ Can view performance metrics
- ✅ Stop-loss protects capital

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Groq API downtime** | Medium | High | Add fallback explanations |
| **Database schema changes** | Low | Medium | Use migrations, version control |
| **Prediction service unavailable** | Low | High | Return cached recommendations |
| **Performance degradation** | Medium | Medium | Implement caching, rate limiting |
| **Security vulnerability** | Low | Critical | Follow security checklist, code review |

---

## Next Steps

### Immediate Actions (Today)
1. ✅ Review this analysis with team
2. ✅ Prioritize blockers vs enhancements
3. ✅ Assign tasks to developers
4. ✅ Set up test database

### This Week
1. Complete database persistence (Blocker 1)
2. Wire API endpoints (Blocker 2)
3. Finish data aggregator (Blocker 3)
4. Basic manual testing

### Next Week
1. Add enhancements (performance chart, rate limiting)
2. Write comprehensive tests
3. Code review and refactoring
4. Staging deployment

---

## Contact

**Questions about this plan?**
- Technical: Review with development team
- Priority: Discuss with product owner
- Timeline: Adjust based on team capacity

**Document Version**: 1.0  
**Created**: February 8, 2026  
**Next Review**: After Day 3 completion
