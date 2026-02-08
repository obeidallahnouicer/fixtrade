"""
Example usage of the AI Decision Agent.

This script demonstrates how to use the AI module for:
- Profile assessment
- Portfolio creation
- Recommendation generation
- Trade execution
- Performance tracking
"""

import asyncio
from decimal import Decimal

from app.ai import DecisionAgent, RiskProfile, UserProfileManager
from app.ai.recommendations import RecommendationEngine


async def example_scenario_1_beginner():
    """
    Scénario 1 : Ahmed, l'investisseur débutant
    
    - Évalue son profil via questionnaire
    - Crée un portefeuille modéré avec 5000 TND
    - Obtient des recommandations personnalisées
    - Exécute un premier achat
    """
    print("=" * 60)
    print("Scénario 1 : L'Investisseur Débutant (Ahmed)")
    print("=" * 60)
    
    # Étape 1 : Évaluation du profil
    print("\n[1] Évaluation du profil via questionnaire...")
    
    profile_manager = UserProfileManager()
    questionnaire = {
        "age": 28,
        "investment_horizon": 5,  # 5 ans
        "income_stability": "high",
        "investment_experience": "beginner",
        "loss_tolerance": 3,  # Modéré (échelle 1-5)
        "financial_goals": "growth"
    }
    
    recommended_profile = profile_manager.recommend_profile(questionnaire)
    print(f"   → Profil recommandé : {recommended_profile.value}")
    
    characteristics = profile_manager.get_characteristics(recommended_profile)
    print(f"   → Max position : {characteristics.max_position_size:.1%}")
    print(f"   → Max actions : {characteristics.max_equity_allocation:.1%}")
    print(f"   → Stop-loss : {characteristics.stop_loss_threshold:.1%}")
    
    # Étape 2 : Création du portefeuille
    print("\n[2] Création du portefeuille...")
    
    agent = DecisionAgent(
        risk_profile=recommended_profile,
        initial_capital=5000.0  # 5000 TND
    )
    
    print(f"   → Capital initial : {agent.portfolio.initial_capital} TND")
    print(f"   → Profil de risque : {agent.risk_profile.value}")
    
    # Étape 3 : Obtenir des recommandations (simulé)
    print("\n[3] Génération des recommandations...")
    print("   → Analyse des 50 valeurs BVMT...")
    print("   → Top 5 opportunités identifiées :")
    
    # Simulation de recommandations
    mock_recommendations = [
        ("AMEN", "BUY", "HIGH", 2.5),
        ("ATTIJARI", "BUY", "MEDIUM", 1.8),
        ("TUNISIE_TELECOM", "HOLD", "MEDIUM", 0.5),
        ("SFBT", "SELL", "LOW", -1.2),
        ("BNA", "BUY", "HIGH", 3.1)
    ]
    
    for i, (symbol, signal, strength, pred_return) in enumerate(mock_recommendations, 1):
        print(f"   {i}. {symbol:20s} {signal:12s} ({strength:8s}) {pred_return:+.1f}%")
    
    # Étape 4 : Exécuter un achat
    print("\n[4] Exécution d'un achat : AMEN Bank...")
    
    # Simuler un prix actuel
    current_price = 12.50
    quantity = 100
    
    result = await agent.execute_trade(
        session=None,  # Pas de DB pour l'exemple
        symbol="AMEN",
        action="buy",
        quantity=quantity,
        price=current_price,
        generate_explanation=False
    )
    
    if result["success"]:
        print(f"   ✓ {result['message']}")
        print(f"   → Coût total : {quantity * current_price:.2f} TND")
        print(f"   → Cash restant : {agent.portfolio.cash_balance:.2f} TND")
    else:
        print(f"   ✗ Échec : {result['message']}")
    
    # Étape 5 : Consulter le portefeuille
    print("\n[5] État du portefeuille...")
    
    snapshot = agent.get_portfolio_snapshot()
    print(f"   → Valeur totale : {snapshot['total_value']:.2f} TND")
    print(f"   → Liquidités : {snapshot['cash_balance']:.2f} TND")
    print(f"   → Actions : {snapshot['equity_value']:.2f} TND ({snapshot['equity_allocation']:.1%})")
    print(f"   → Positions : {len(snapshot['positions'])}")
    
    for pos in snapshot['positions']:
        print(f"      • {pos['symbol']} : {pos['quantity']} actions @ {pos['purchase_price']:.3f} TND")


async def example_scenario_2_trader():
    """
    Scénario 2 : Leila, la trader avertie
    
    - Utilise un profil agressif
    - Réagit à une alerte d'anomalie
    - Exécute une stratégie de timing
    """
    print("\n" + "=" * 60)
    print("Scénario 2 : Le Trader Averti (Leila)")
    print("=" * 60)
    
    # Création avec profil agressif
    print("\n[1] Initialisation profil agressif...")
    
    agent = DecisionAgent(
        risk_profile=RiskProfile.AGGRESSIVE,
        initial_capital=20000.0
    )
    
    chars = agent.profile_manager.get_characteristics(RiskProfile.AGGRESSIVE)
    print(f"   → Max position : {chars.max_position_size:.1%}")
    print(f"   → Max actions : {chars.max_equity_allocation:.1%}")
    print(f"   → Stop-loss : {chars.stop_loss_threshold:.1%}")
    
    # Simulation d'une alerte anomalie
    print("\n[2] ⚠️  Alerte : Anomalie détectée sur SFBT !")
    print("   → Volume : +800% (10x la moyenne)")
    print("   → Prix : +12% sans news significative")
    print("   → Sentiment : Neutre (aucun article récent)")
    print("   → Suspicion : Possible manipulation")
    
    print("\n[3] Décision : ATTENDRE 24h pour confirmation")
    print("   → Volatilité élevée prévue")
    print("   → Recommandation : Observer avant d'agir")
    
    print("\n[4] 24h plus tard : Prix stabilisé à 8.50 TND")
    print("   → Exécution d'un achat prudent...")
    
    result = await agent.execute_trade(
        session=None,
        symbol="SFBT",
        action="buy",
        quantity=200,
        price=8.50,
        generate_explanation=False
    )
    
    if result["success"]:
        print(f"   ✓ {result['message']}")


def example_metrics_calculation():
    """
    Exemple de calcul de métriques de performance.
    """
    print("\n" + "=" * 60)
    print("Calcul de Métriques de Performance")
    print("=" * 60)
    
    # Créer un agent avec quelques trades simulés
    agent = DecisionAgent(
        risk_profile=RiskProfile.MODERATE,
        initial_capital=10000.0
    )
    
    # Simuler des prix mis à jour
    print("\n[1] Simulation de positions...")
    
    # Achats
    agent.portfolio.buy("AMEN", 100, 12.50)
    agent.portfolio.buy("ATTIJARI", 50, 25.00)
    agent.portfolio.buy("BNA", 200, 5.00)
    
    # Mise à jour des prix (gains)
    agent.update_market_prices({
        "AMEN": 15.00,     # +20%
        "ATTIJARI": 26.50, # +6%
        "BNA": 5.50        # +10%
    })
    
    print("   Positions :")
    for symbol, pos in agent.portfolio.positions.items():
        print(f"   • {symbol:15s} : {pos.quantity:3d} actions, P&L: {pos.unrealized_pnl:+.2f} TND ({pos.unrealized_pnl_pct:+.1f}%)")
    
    # Vendre une position (réaliser un gain)
    agent.portfolio.sell("AMEN", 100, 15.00)
    
    print("\n[2] Calcul des métriques...")
    
    metrics = agent.get_performance_metrics()
    
    print(f"\n   Valeur Totale     : {metrics['total_value']:.2f} TND")
    print(f"   Rendement         : {metrics['total_return']:+.2f}%")
    print(f"   ROI               : {metrics['roi']:+.2f}%")
    print(f"   Sharpe Ratio      : {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown      : {metrics['max_drawdown']:.2f}%")
    print(f"   Volatilité        : {metrics['volatility']:.2f}%")
    print(f"\n   Trades Totaux     : {metrics['total_trades']}")
    print(f"   Trades Gagnants   : {metrics['winning_trades']}")
    print(f"   Trades Perdants   : {metrics['losing_trades']}")
    print(f"   Taux de Réussite  : {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor     : {metrics['profit_factor']:.2f}")
    
    # Interprétation
    print("\n[3] Interprétation :")
    
    if metrics['sharpe_ratio'] > 2.0:
        print("   ✓ Excellente performance ajustée au risque !")
    elif metrics['sharpe_ratio'] > 1.0:
        print("   ✓ Bonne performance ajustée au risque")
    else:
        print("   ⚠️  Performance moyenne, amélioration possible")
    
    if metrics['win_rate'] > 70:
        print("   ✓ Excellent taux de réussite des trades")
    elif metrics['win_rate'] > 50:
        print("   ✓ Bon taux de réussite")
    else:
        print("   ⚠️  Taux de réussite faible, réviser la stratégie")


def example_stop_loss():
    """
    Exemple de gestion des stop-loss.
    """
    print("\n" + "=" * 60)
    print("Gestion des Stop-Loss")
    print("=" * 60)
    
    agent = DecisionAgent(
        risk_profile=RiskProfile.CONSERVATIVE,  # Stop-loss à 5%
        initial_capital=10000.0
    )
    
    print("\n[1] Achat initial...")
    agent.portfolio.buy("TEST_STOCK", 100, 10.00)
    print("   ✓ 100 TEST_STOCK @ 10.00 TND")
    
    print("\n[2] Simulation de baisse du prix...")
    
    scenarios = [
        (9.70, "Baisse de -3%"),
        (9.50, "Baisse de -5% → Stop-loss déclenché !"),
        (9.00, "Baisse de -10%")
    ]
    
    for price, description in scenarios:
        agent.update_market_prices({"TEST_STOCK": price})
        triggered = agent.portfolio.check_stop_losses()
        
        print(f"\n   Prix : {price:.2f} TND ({description})")
        
        if triggered:
            print(f"   ⚠️  STOP-LOSS DÉCLENCHÉ pour : {', '.join(triggered)}")
            print("   → Vente automatique recommandée")
            
            # Exécuter le stop-loss
            executed = agent.check_and_handle_stop_losses({"TEST_STOCK": price})
            
            if executed:
                for trade in executed:
                    print(f"   ✓ Vendu {trade['quantity']} @ {trade['price']:.2f} TND")
                    print(f"   → P&L : {trade['profit_loss']:+.2f} TND")
            break
        else:
            print("   ✓ Position maintenue")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EXEMPLES D'UTILISATION AI MODULE" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Scénario 1 : Débutant
    await example_scenario_1_beginner()
    
    # Scénario 2 : Trader
    await example_scenario_2_trader()
    
    # Métriques
    example_metrics_calculation()
    
    # Stop-loss
    example_stop_loss()
    
    print("\n" + "=" * 60)
    print("Tous les exemples terminés !")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
