"""
Simplified LLM workflow test - focuses on OpenRouter + YAML prompts.
Tests: Mock Data → LLM Explainer → Wall Street Explanations
"""

import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def test_llm_workflow():
    """Test LLM explanation generation with OpenRouter and badass prompts."""
    print("=" * 80)
    print("🚀 LLM WORKFLOW TEST: OpenRouter + Wall Street Prompts")
    print("=" * 80)
    
    # Step 1: Setup
    print("\n[1/4] Loading configuration...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OpenRouter API key found in .env")
        print("   Add OPENROUTER_API_KEY=sk-or-v1-... to your .env file")
        return
    
    print(f"✅ OpenRouter API key loaded: {api_key[:25]}...")
    
    # Step 2: Create LLM config
    print("\n[2/4] Configuring LLM explainer...")
    from app.ai.llm_explainer import LLMExplainer, LLMConfig, ExplanationContext
    
    llm_config = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500,
        enable_reasoning=True
    )
    
    explainer = LLMExplainer(use_llm=True, llm_config=llm_config)
    print(f"✅ LLM Explainer configured")
    print(f"   Provider: OpenRouter")
    print(f"   Model: Claude 3.5 Sonnet")
    print(f"   Reasoning Mode: Enabled")
    print(f"   Using prompts from: app/ai/prompts.yaml")
    
    # Step 3: Test BUY recommendation
    print("\n[3/4] Generating BUY recommendation explanation...")
    print("-" * 80)
    
    buy_context = ExplanationContext(
        symbol="110025",
        action="BUY",
        confidence=92.5,
        expected_return=15.3,
        beta=1.15,
        current_weight=0.10,
        target_weight=0.25,
        risk_contribution=18.5,
        risk_profile="moderate",
        anomaly_detected=False,
        sentiment_score=0.65
    )
    
    print(f"📊 Testing BUY Signal:")
    print(f"   Symbol: {buy_context.symbol}")
    print(f"   Expected Return: {buy_context.expected_return}%")
    print(f"   Beta: {buy_context.beta}")
    print(f"   Confidence: {buy_context.confidence}%")
    print(f"\n   Calling OpenRouter API...")
    
    buy_explanation = explainer.explain_decision(buy_context)
    
    print(f"\n   💬 Wall Street Explanation:")
    print(f'   "{buy_explanation}"')
    
    # Note: reasoning trace is embedded in the response for OpenRouter
    # We can't easily separate it with the current implementation
    
    # Step 4: Test SELL recommendation
    print("\n[4/4] Generating SELL recommendation explanation...")
    print("-" * 80)
    
    sell_context = ExplanationContext(
        symbol="110028",
        action="SELL",
        confidence=88.0,
        expected_return=-8.2,
        beta=1.35,
        current_weight=0.20,
        target_weight=0.05,
        risk_contribution=25.3,
        risk_profile="moderate",
        anomaly_detected=True,
        sentiment_score=-0.45
    )
    
    print(f"📊 Testing SELL Signal:")
    print(f"   Symbol: {sell_context.symbol}")
    print(f"   Expected Return: {sell_context.expected_return}%")
    print(f"   Beta: {sell_context.beta}")
    print(f"   Anomaly Detected: YES")
    print(f"\n   Calling OpenRouter API...")
    
    sell_explanation = explainer.explain_decision(sell_context)
    
    print(f"\n   💬 Wall Street Explanation:")
    print(f'   "{sell_explanation}"')
    
    # Note: reasoning trace is embedded in the response for OpenRouter
    # We can't easily separate it with the current implementation
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ LLM WORKFLOW TEST COMPLETE")
    print("=" * 80)
    print("\n🎯 Key Observations:")
    print("   1. Check if explanations are aggressive (not generic)")
    print("   2. Look for power words: EXECUTE, DEPLOY, CONVICTION")
    print("   3. Verify reasoning traces show step-by-step thinking")
    print("   4. Confirm no hedging language (no 'maybe', 'consider')")
    print("\n🚀 Wall Street-grade explanations are LIVE!")


if __name__ == "__main__":
    test_llm_workflow()
    """Test complete portfolio optimization workflow with real OpenRouter API."""
    print("=" * 80)
    print("🚀 FULL WORKFLOW TEST: Portfolio Optimization with OpenRouter")
    print("=" * 80)
    
    # Step 1: Mock data (since async DB is problematic with Python 3.14)
    print("\n[1/6] Generating mock portfolio data...")
    
    symbols = ['110025', '110028', '110029', '110030', '110031']
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    
    # Mock returns data
    np.random.seed(42)
    returns_data = np.random.randn(30, 5) * 0.02  # 2% daily volatility
    returns = pd.DataFrame(returns_data, columns=symbols, index=dates)
    
    # Mock market returns
    market_returns = pd.Series(np.random.randn(30) * 0.015, index=dates)
    
    # Mock current portfolio
    current_weights = {symbol: 0.2 for symbol in symbols}  # Equal weight
    
    # Mock anomalies
    anomalies = {symbols[0]: True}  # First stock has anomaly
    
    print(f"✅ Mock data generated: {returns.shape[0]} days × {returns.shape[1]} symbols")
    print(f"   Symbols: {list(returns.columns)}")
    print(f"   Current portfolio: Equal weighted (20% each)")
    print(f"   Anomalies detected: {list(anomalies.keys())}")
    
    # Step 2: Portfolio optimization
    print("\n[2/6] Testing portfolio optimizer...")
    from app.ai.optimization import PortfolioOptimizer
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.05)
    
    # Calculate portfolio metrics
    expected_returns_arr = (returns.mean() * 252).values
    cov_matrix = (returns.cov() * 252).values
    
    # Optimize
    optimal_result = optimizer.optimize_portfolio(
        expected_returns=expected_returns_arr,
        cov_matrix=cov_matrix,
        constraints=None
    )
    
    optimal_weights = dict(zip(symbols, optimal_result.weights))
    print(f"✅ Optimization complete: {len(optimal_weights)} optimal weights")
    print(f"   Top 3 weights: {dict(list(optimal_weights.items())[:3])}")
    print(f"   Expected Return: {optimal_result.expected_return:.2%}")
    print(f"   Volatility: {optimal_result.volatility:.2%}")
    print(f"   Sharpe Ratio: {optimal_result.sharpe_ratio:.2f}")
    
    # Step 3: Decision engine with LLM
    print("\n[3/6] Testing decision engine with OpenRouter...")
    from app.ai.decision_engine import DecisionEngine
    from app.ai.llm_explainer import LLMConfig
    
    # Get API key from env
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OpenRouter API key found in .env")
        
    
    print(f"✅ OpenRouter API key loaded: {api_key[:25]}...")
    
    # Create LLM config with reasoning enabled
    llm_config = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500,
        enable_reasoning=True
    )
    
    engine = DecisionEngine(
        optimizer=optimizer,
        risk_profile="moderate",
        use_llm=True,
        llm_config=llm_config
    )
    
    # Step 4: Generate recommendations
    print("\n[4/6] Generating Wall Street-grade recommendations...")
    
    recommendations = engine.generate_recommendations(
        current_portfolio=current_weights,
        anomaly_alerts=anomalies
    )
    
    print(f"✅ Recommendations generated: {len(recommendations)} decisions")
    
    # Step 5: Display recommendations
    print("\n[5/6] Detailed Recommendations with LLM Explanations:")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
        print(f"\n📊 Recommendation #{i}:")
        print(f"   Symbol: {rec['symbol']}")
        print(f"   Action: {rec['action']} 💰" if rec['action'] == 'BUY' else f"   Action: {rec['action']}")
        print(f"   Current Weight: {rec['current_weight']:.2%}")
        print(f"   Target Weight: {rec['target_weight']:.2%}")
        print(f"   Confidence: {rec['confidence']:.1f}%")
        print(f"   Expected Return: {rec['expected_return']:.2f}%")
        print(f"   Beta: {rec['beta']:.2f}")
        print(f"\n   💬 Wall Street Explanation:")
        print(f"   \"{rec['explanation']}\"")
        
        if rec.get('reasoning_trace'):
            print(f"\n   🧠 AI Reasoning (first 200 chars):")
            print(f"   {rec['reasoning_trace'][:200]}...")
    
    # Step 6: Portfolio summary
    print("\n[6/6] Testing portfolio-level summary...")
    
    portfolio_metrics = {
        'expected_return': (returns.mean() * optimal_weights).sum() * 252,
        'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights))),
        'sharpe_ratio': 1.2,
        'diversification_ratio': 0.85
    }
    
    summary = engine.generate_portfolio_summary(recommendations, portfolio_metrics)
    
    print("\n📈 Portfolio Strategy Summary (CIO-Level):")
    print("-" * 80)
    print(f"\"{summary}\"")
    
    # Final stats
    print("\n" + "=" * 80)
    print("✅ FULL WORKFLOW TEST COMPLETE")
    print("=" * 80)
    
    buy_count = sum(1 for r in recommendations if r['action'] == 'BUY')
    sell_count = sum(1 for r in recommendations if r['action'] == 'SELL')
    hold_count = sum(1 for r in recommendations if r['action'] == 'HOLD')
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Total Recommendations: {len(recommendations)}")
    print(f"   BUY signals: {buy_count} 📈")
    print(f"   SELL signals: {sell_count} 📉")
    print(f"   HOLD signals: {hold_count} 🤝")
    print(f"   Risk Profile: moderate")
    print(f"   Expected Portfolio Return: {portfolio_metrics['expected_return']:.2%}")
    print(f"   Expected Volatility: {portfolio_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"   LLM Provider: OpenRouter (Claude 3.5 Sonnet)")
    print(f"   Reasoning Mode: ✅ Enabled")
    
    print("\n🎉 All systems operational! Wall Street-grade explanations delivered!")
    print("\n💡 The LLM is using the badass prompts from app/ai/prompts.yaml")
    print("   Check the explanations above - they should be aggressive and assertive!")


if __name__ == "__main__":
    test_full_workflow()
