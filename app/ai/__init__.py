"""
AI Decision Agent and Portfolio Management Module.

This module implements an intelligent trading assistant that:
- Manages user risk profiles (conservative/moderate/aggressive)
- Aggregates insights from prediction, sentiment, and anomaly modules
- Simulates portfolio performance with virtual capital
- Provides explainable recommendations using Groq AI
- Tracks metrics: ROI, Sharpe Ratio, Max Drawdown

Usage:
    from app.ai import DecisionAgent, PortfolioManager
    
    agent = DecisionAgent(risk_profile="moderate")
    recommendations = await agent.get_daily_recommendations(top_n=5)
"""

from app.ai.agent import DecisionAgent
from app.ai.portfolio import PortfolioManager
from app.ai.profile import RiskProfile, UserProfileManager
from app.ai.recommendations import RecommendationEngine
from app.ai.explainability import ExplanationGenerator

__all__ = [
    "DecisionAgent",
    "PortfolioManager",
    "RiskProfile",
    "UserProfileManager",
    "RecommendationEngine",
    "ExplanationGenerator",
]
