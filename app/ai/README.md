# AI Decision Agent & Portfolio Management

## Vue d'ensemble

Le module AI implémente un agent de décision intelligent pour le trading et la gestion de portefeuille sur la Bourse de Valeurs Mobilières de Tunis (BVMT).

### Fonctionnalités Principales

1. **Gestion des Profils Utilisateur**
   - 3 profils de risque : Conservateur / Modéré / Agressif
   - Questionnaire d'évaluation automatique
   - Paramètres adaptés par profil

2. **Agrégation Intelligente**
   - Intégration des prédictions de prix
   - Analyse de sentiment (NLP)
   - Détection d'anomalies
   - Évaluation de liquidité
   - Analyse de volume

3. **Simulation de Portefeuille**
   - Capital virtuel configurable (défaut: 10,000 TND)
   - Suivi temps réel des positions
   - Gestion du risque (stop-loss)
   - Respect des limites par profil

4. **Métriques de Performance**
   - ROI (Return on Investment)
   - Sharpe Ratio (rendement ajusté au risque)
   - Maximum Drawdown
   - Volatilité annualisée
   - Taux de réussite des trades

5. **Explainability (IA Générative)**
   - Explications en langage naturel via Groq API
   - Justification de chaque recommandation
   - Contexte personnalisé au portefeuille

## Architecture

```
app/ai/
├── __init__.py              # Exports du module
├── config.py                # Configuration AI (Groq, seuils)
├── profile.py               # Gestion profils de risque
├── portfolio.py             # Moteur de simulation portefeuille
├── metrics.py               # Calculateur de métriques
├── rules.py                 # Système de règles de décision
├── aggregator.py            # Agrégation données multi-sources
├── recommendations.py       # Moteur de recommandations
├── explainability.py        # Génération d'explications (Groq)
├── agent.py                 # Agent de décision principal
└── router.py                # Endpoints FastAPI
```

## Configuration

### Variables d'Environnement (.env)

Ajouter à votre fichier `.env` :

```bash
# Groq AI (pour explainability)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1024
GROQ_TEMPERATURE=0.7

# Portfolio Defaults
DEFAULT_INITIAL_CAPITAL=10000.0
DEFAULT_RISK_PROFILE=moderate

# Risk Thresholds
CONSERVATIVE_MAX_POSITION_SIZE=0.10
MODERATE_MAX_POSITION_SIZE=0.15
AGGRESSIVE_MAX_POSITION_SIZE=0.25

MIN_CONFIDENCE_SCORE=0.65
ANOMALY_SEVERITY_THRESHOLD=0.75
```

### Installation

```bash
# Installer la dépendance Groq
pip install groq

# Ou via requirements.txt
pip install -r requirements.txt
```

## Utilisation

### 1. API REST

#### Déterminer le Profil Utilisateur

```http
POST /api/v1/ai/profile/questionnaire
Content-Type: application/json

{
  "age": 28,
  "investment_horizon": 5,
  "income_stability": "high",
  "investment_experience": "beginner",
  "loss_tolerance": 3,
  "financial_goals": "growth"
}
```

**Réponse :**
```json
{
  "recommended_profile": "moderate",
  "characteristics": {
    "max_position_size": 0.15,
    "max_equity_allocation": 0.70,
    "stop_loss_threshold": 0.08,
    "min_holding_days": 3,
    "preferred_liquidity": "medium",
    "description": "Profil modéré : Équilibre entre risque et rendement..."
  }
}
```

#### Créer un Portefeuille

```http
POST /api/v1/ai/portfolio/create
Content-Type: application/json

{
  "risk_profile": "moderate",
  "initial_capital": 10000.0
}
```

#### Obtenir les Recommandations du Jour

```http
GET /api/v1/ai/recommendations?portfolio_id=default&top_n=10
```

**Réponse :**
```json
[
  {
    "symbol": "AMEN",
    "signal": "BUY",
    "strength": "HIGH",
    "explanation": "**ACHAT** recommandé pour AMEN Bank...",
    "predicted_return": 2.5,
    "confidence": 0.85,
    "current_price": 12.50,
    "timestamp": "2026-02-08T10:30:00"
  }
]
```

#### Expliquer une Recommandation

```http
GET /api/v1/ai/recommendations/AMEN/explain?portfolio_id=default
```

**Réponse :**
```json
{
  "symbol": "AMEN",
  "explanation": "AMEN Bank présente une opportunité d'achat intéressante basée sur trois facteurs clés. Premièrement, notre modèle prévoit une hausse de 2.5% avec une confiance élevée de 85%. Deuxièmement, l'analyse de sentiment des articles récents est très positive (score: +0.72), suggérant un momentum favorable. Troisièmement, la liquidité est élevée, minimisant le risque d'exécution. Attention : surveillez le volume pour confirmer la tendance."
}
```

#### Exécuter un Trade

```http
POST /api/v1/ai/portfolio/default/trade
Content-Type: application/json

{
  "symbol": "AMEN",
  "action": "buy",
  "quantity": 100,
  "price": 12.50,
  "generate_explanation": true
}
```

#### Consulter le Portefeuille

```http
GET /api/v1/ai/portfolio/default/snapshot
```

**Réponse :**
```json
{
  "portfolio_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-08T10:30:00",
  "total_value": 10250.00,
  "cash_balance": 8750.00,
  "equity_value": 1500.00,
  "equity_allocation": 0.146,
  "positions": [
    {
      "symbol": "AMEN",
      "quantity": 100,
      "purchase_price": 12.50,
      "current_price": 15.00,
      "current_value": 1500.00,
      "unrealized_pnl": 250.00,
      "unrealized_pnl_pct": 20.00
    }
  ]
}
```

#### Métriques de Performance

```http
GET /api/v1/ai/portfolio/default/performance
```

**Réponse :**
```json
{
  "total_value": 10250.00,
  "total_return": 2.50,
  "roi": 2.50,
  "sharpe_ratio": 1.45,
  "max_drawdown": -1.20,
  "volatility": 12.50,
  "total_trades": 5,
  "winning_trades": 4,
  "losing_trades": 1,
  "win_rate": 80.00,
  "profit_factor": 3.50,
  "days_active": 30,
  "annualized_return": 31.50
}
```

### 2. Utilisation Programmatique

```python
from app.ai import DecisionAgent, RiskProfile

# Créer un agent
agent = DecisionAgent(
    risk_profile=RiskProfile.MODERATE,
    initial_capital=10000.0
)

# Obtenir des recommandations
recommendations = await agent.get_daily_recommendations(
    session=db_session,
    top_n=10
)

# Exécuter un trade
result = await agent.execute_trade(
    session=db_session,
    symbol="AMEN",
    action="buy",
    quantity=100,
    price=12.50
)

# Consulter les métriques
metrics = agent.get_performance_metrics()
print(f"ROI: {metrics['roi']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## Scénarios d'Usage

### Scénario 1 : L'Investisseur Débutant (Ahmed)

1. **Questionnaire de profil** → Profil "Modéré" recommandé
2. **Création de portefeuille** avec 5000 TND
3. **Recommandations du jour** → 5 valeurs sûres suggérées
4. **Consultation détaillée** → "Expliquer" pour Tunisie Télécom
5. **Exécution d'achat** → 100 actions avec confirmation
6. **Suivi performance** → Dashboard temps réel

### Scénario 2 : Le Trader Averti (Leila)

1. **Alertes anomalies** → Volume anormal sur SFBT détecté
2. **Analyse détaillée** → Graphique + sentiment + prévision
3. **Décision éclairée** → Attente 24h recommandée
4. **Trade stratégique** → Entrée en position après stabilisation
5. **Tracking performance** → Analyse continue de la performance

### Scénario 3 : Le Régulateur (CMF)

1. **Alerte anomalie** → Variation +12% sans news significative
2. **Dashboard surveillance** → Volume 10x, sentiment neutre
3. **Analyse suspicion** → Possible manipulation
4. **Rapport détaillé** → Timeline, ordres, métriques

## Système de Règles

Le moteur de décision évalue plusieurs signaux :

### Signaux Analysés

1. **Prédiction de Prix**
   - Hausse > 5% : Score +2.0
   - Hausse > 2% : Score +1.0
   - Baisse < -5% : Score -2.0
   - Baisse < -2% : Score -1.0

2. **Sentiment**
   - Très positif (> 0.5) : Score +1.5
   - Positif (> 0.2) : Score +0.75
   - Négatif (< -0.2) : Score -0.75
   - Très négatif (< -0.5) : Score -1.5

3. **Anomalies**
   - Sévère (> 0.75) : Score -2.0
   - Modérée : Score -1.0

4. **Liquidité**
   - Faible (profil conservateur) : Score -1.5
   - Élevée (tous profils) : Score +0.5

5. **Volume**
   - Élevé (> 100%) : Score +0.5
   - Faible (< -50%) : Score -0.5

### Signaux de Sortie

- **Score ≥ 3.0** : STRONG_BUY
- **Score ≥ 1.5** : BUY
- **Score ≤ -3.0** : STRONG_SELL
- **Score ≤ -1.5** : SELL
- **Autre** : HOLD

### Ajustements par Profil

- **Conservateur** : Downgrade si anomalies, exige confiance > 75%
- **Modéré** : Équilibre standard
- **Agressif** : Upgrade si rendement > 7%

## Métriques de Performance

### ROI (Return on Investment)
```
ROI = (Valeur Actuelle - Capital Initial) / Capital Initial × 100
```

### Sharpe Ratio
```
Sharpe = (Rendement Moyen - Taux Sans Risque) / Volatilité
```
- > 2.0 : Excellent
- > 1.0 : Bon
- < 1.0 : Moyen

### Maximum Drawdown
Pire déclin sommet-creux du portefeuille (%)

### Volatilité
Écart-type annualisé des rendements

## Intégration avec Modules Existants

### Prediction Module
```python
# aggregator.py utilise prediction.inference
prediction = await prediction_service.predict_price(symbol)
volume = await prediction_service.predict_volume(symbol)
liquidity = await prediction_service.predict_liquidity(symbol)
```

### NLP Module
```python
# aggregator.py utilise app.nlp.sentiment
from app.nlp.sentiment import SentimentAnalyzer
analyzer = SentimentAnalyzer()
score = analyzer.analyze(article_text)
```

### Database
```python
# Récupération des données
- stock_prices : Prix historiques
- price_predictions : Prévisions
- sentiment_scores : Scores de sentiment
- anomaly_alerts : Alertes d'anomalies
- portfolios : État des portefeuilles
- portfolio_positions : Positions actives
```

## Tests

```bash
# Tests unitaires
pytest app/ai/tests/

# Test de l'API
curl http://localhost:8000/api/v1/ai/status

# Exemple de test
pytest app/ai/tests/test_portfolio.py -v
```

## Prochaines Étapes (Reinforcement Learning)

Le système actuel utilise un **système de règles** sophistiqué. Les évolutions futures incluront :

1. **RL Agent** : Agent d'apprentissage par renforcement (PPO/DQN)
2. **Training Pipeline** : Entraînement sur données historiques
3. **Reward Function** : Sharpe Ratio + conformité profil
4. **Auto-adaptation** : Apprentissage continu des stratégies

## Support

- Documentation : `/docs` (FastAPI Swagger)
- Logs : Configurés via `app.shared.logging`
- Monitoring : Métriques de performance en temps réel

## Références

- Groq AI : https://groq.com
- Portfolio Metrics : https://www.investopedia.com/terms/s/sharperatio.asp
- Risk Management : https://www.cmf.tn/
