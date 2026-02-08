"""
Recommendation Engine.

Generates daily trading recommendations by:
1. Fetching candidate stocks
2. Aggregating signals for each
3. Evaluating with rule-based system
4. Ranking and filtering by risk profile
5. Generating explanations

Produces actionable recommendations with clear rationale.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.config import ai_settings
from app.ai.profile import RiskProfile
from app.ai.rules import RuleBasedEngine, Signal, SignalStrength
from app.ai.aggregator import DataAggregator
from app.ai.explainability import ExplanationGenerator

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single stock recommendation."""
    symbol: str
    signal: Signal
    strength: SignalStrength
    explanation: str
    score: float  # Internal priority score
    
    # Supporting data
    predicted_return: Optional[float] = None
    confidence: Optional[float] = None
    current_price: Optional[float] = None
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "strength": self.strength.value,
            "explanation": self.explanation,
            "predicted_return": self.predicted_return,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "timestamp": self.timestamp.isoformat()
        }


class RecommendationEngine:
    """
    Generates daily stock recommendations.
    
    Workflow:
    1. Select candidate stocks (top movers, user watchlist, etc.)
    2. Aggregate signals for each stock
    3. Evaluate with rule-based system
    4. Generate explanations
    5. Rank and filter results
    """
    
    def __init__(
        self,
        risk_profile: RiskProfile = RiskProfile.MODERATE
    ):
        """Initialize recommendation engine."""
        self.risk_profile = risk_profile
        self.aggregator = DataAggregator()
        self.rule_engine = RuleBasedEngine(risk_profile)
        self.explainer = ExplanationGenerator()
        
        logger.info(f"RecommendationEngine initialized with profile: {risk_profile}")
    
    async def generate_recommendations(
        self,
        session: AsyncSession,
        top_n: int = 10,
        symbols: Optional[List[str]] = None,
        prediction_service = None
    ) -> List[Recommendation]:
        """
        Generate top N recommendations.
        
        Args:
            session: Database session
            top_n: Number of recommendations to return
            symbols: Specific symbols to analyze (None = auto-select)
            prediction_service: Prediction inference service
        
        Returns:
            List of Recommendation objects, ranked by priority
        """
        logger.info(f"Generating {top_n} recommendations for profile {self.risk_profile}")
        
        # Step 1: Get candidate stocks
        if symbols is None:
            symbols = await self._select_candidates(session, limit=50)
        
        if not symbols:
            logger.warning("No candidate stocks found")
            return []
        
        logger.info(f"Analyzing {len(symbols)} candidate stocks")
        
        # Step 2: Evaluate each stock
        raw_recommendations = []
        
        for symbol in symbols:
            try:
                # Aggregate signals
                signals = await self.aggregator.get_signals(
                    symbol, session, prediction_service
                )
                
                # Evaluate with rules
                signal, strength, reasons = self.rule_engine.evaluate(signals)
                
                # Skip HOLD with low confidence for now
                if signal == Signal.HOLD and strength == SignalStrength.LOW:
                    continue
                
                # Calculate priority score
                score = self._calculate_score(signal, strength, signals)
                
                # Generate explanation (basic for now, detailed on demand)
                explanation = self.explainer._generate_fallback_explanation(
                    symbol, signal, strength, signals, reasons
                )
                
                recommendation = Recommendation(
                    symbol=symbol,
                    signal=signal,
                    strength=strength,
                    explanation=explanation,
                    score=score,
                    predicted_return=signals.predicted_return,
                    confidence=signals.confidence_score,
                    current_price=signals.current_price
                )
                
                raw_recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Step 3: Rank and filter
        ranked = self._rank_recommendations(raw_recommendations)
        filtered = ranked[:top_n]
        
        logger.info(f"Generated {len(filtered)} recommendations")
        
        return filtered
    
    async def get_recommendation_with_explanation(
        self,
        symbol: str,
        session: AsyncSession,
        prediction_service = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Recommendation]:
        """
        Get detailed recommendation with AI-generated explanation.
        
        Args:
            symbol: Stock symbol
            session: Database session
            prediction_service: Prediction service
            user_context: User portfolio/preferences
        
        Returns:
            Recommendation with detailed explanation
        """
        try:
            # Aggregate signals
            signals = await self.aggregator.get_signals(
                symbol, session, prediction_service
            )
            
            # Evaluate
            signal, strength, reasons = self.rule_engine.evaluate(signals)
            
            # Generate detailed explanation using Groq
            explanation = await self.explainer.explain_recommendation(
                symbol=symbol,
                signal=signal,
                strength=strength,
                signals=signals,
                reasons=reasons,
                user_context=user_context
            )
            
            score = self._calculate_score(signal, strength, signals)
            
            recommendation = Recommendation(
                symbol=symbol,
                signal=signal,
                strength=strength,
                explanation=explanation,
                score=score,
                predicted_return=signals.predicted_return,
                confidence=signals.confidence_score,
                current_price=signals.current_price
            )
            
            logger.info(f"Generated detailed recommendation for {symbol}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            return None
    
    async def _select_candidates(
        self,
        session: AsyncSession,
        limit: int = 50
    ) -> List[str]:
        """
        Select candidate stocks for analysis.
        
        Strategy:
        - Top movers (volume/volatility)
        - Stocks with recent predictions
        - Stocks with recent news/sentiment
        """
        try:
            # For now, get top movers
            candidates = await self.aggregator.get_top_movers(session, limit=limit)
            
            if not candidates:
                # Fallback: get all symbols
                candidates = await self.aggregator.get_all_symbols(session)
            
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Error selecting candidates: {e}")
            return []
    
    def _calculate_score(
        self,
        signal: Signal,
        strength: SignalStrength,
        signals
    ) -> float:
        """
        Calculate priority score for ranking.
        
        Higher score = higher priority
        """
        # Base score from signal
        signal_scores = {
            Signal.STRONG_BUY: 10.0,
            Signal.BUY: 7.0,
            Signal.HOLD: 3.0,
            Signal.SELL: 5.0,
            Signal.STRONG_SELL: 8.0
        }
        
        # Strength multiplier
        strength_multipliers = {
            SignalStrength.VERY_HIGH: 1.5,
            SignalStrength.HIGH: 1.2,
            SignalStrength.MEDIUM: 1.0,
            SignalStrength.LOW: 0.7,
            SignalStrength.VERY_LOW: 0.5
        }
        
        score = signal_scores.get(signal, 0) * strength_multipliers.get(strength, 1.0)
        
        # Boost for high predicted returns
        if signals.predicted_return:
            score += abs(signals.predicted_return) * 0.2
        
        # Boost for high confidence
        if signals.confidence_score:
            score *= (0.7 + 0.3 * signals.confidence_score)
        
        # Penalty for anomalies
        if signals.has_anomaly and signals.anomaly_severity:
            score *= (1.0 - signals.anomaly_severity * 0.3)
        
        return round(score, 2)
    
    def _rank_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Rank recommendations by priority score."""
        return sorted(
            recommendations,
            key=lambda r: r.score,
            reverse=True
        )
