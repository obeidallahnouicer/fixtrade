"""
Rule-Based Decision System.

Implements sophisticated if/else logic for trading decisions based on:
- Price predictions and confidence
- Sentiment analysis
- Anomaly detection
- Liquidity assessment
- Technical indicators
- Risk profile

This serves as the baseline decision engine before
reinforcement learning is added.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import date

from app.ai.config import ai_settings
from app.ai.profile import RiskProfile

logger = logging.getLogger(__name__)


class Signal(str, Enum):
    """Trading signal recommendations."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(str, Enum):
    """Signal strength classification."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class MarketSignals:
    """Aggregated market signals for a stock."""
    symbol: str
    
    # Price prediction
    predicted_price: Optional[float] = None
    predicted_return: Optional[float] = None  # %
    confidence_score: Optional[float] = None
    
    # Sentiment
    sentiment_score: Optional[float] = None  # -1 to 1
    sentiment_label: Optional[str] = None  # positive/negative/neutral
    
    # Anomaly
    has_anomaly: bool = False
    anomaly_severity: Optional[float] = None
    anomaly_type: Optional[str] = None
    
    # Liquidity
    liquidity_tier: Optional[str] = None  # high/medium/low
    liquidity_prob: Optional[float] = None
    
    # Volume
    predicted_volume: Optional[float] = None
    volume_change: Optional[float] = None  # % vs average
    
    # Current state
    current_price: Optional[float] = None
    price_momentum: Optional[float] = None  # Recent trend


class RuleBasedEngine:
    """
    Rule-based trading decision system.
    
    Evaluates multiple signals to produce actionable recommendations
    with configurable risk tolerance.
    """
    
    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE):
        """Initialize with risk profile."""
        self.risk_profile = risk_profile
        logger.info(f"RuleBasedEngine initialized with profile: {risk_profile}")
    
    def evaluate(self, signals: MarketSignals) -> tuple[Signal, SignalStrength, List[str]]:
        """
        Evaluate market signals and generate trading recommendation.
        
        Args:
            signals: Aggregated market signals
        
        Returns:
            (signal, strength, reasons)
        """
        score = 0.0
        reasons = []
        
        # --- Price Prediction Analysis ---
        if signals.predicted_return is not None:
            if signals.predicted_return > 5.0:
                score += 2.0
                reasons.append(f"Forte hausse prévue ({signals.predicted_return:+.1f}%)")
            elif signals.predicted_return > 2.0:
                score += 1.0
                reasons.append(f"Hausse modérée prévue ({signals.predicted_return:+.1f}%)")
            elif signals.predicted_return < -5.0:
                score -= 2.0
                reasons.append(f"Forte baisse prévue ({signals.predicted_return:.1f}%)")
            elif signals.predicted_return < -2.0:
                score -= 1.0
                reasons.append(f"Baisse modérée prévue ({signals.predicted_return:.1f}%)")
            
            # Confidence factor
            if signals.confidence_score is not None:
                confidence_weight = signals.confidence_score
                if signals.confidence_score < ai_settings.min_confidence_score:
                    score *= 0.5
                    reasons.append(f"Confiance faible ({signals.confidence_score:.2f})")
                else:
                    reasons.append(f"Confiance: {signals.confidence_score:.2f}")
        
        # --- Sentiment Analysis ---
        if signals.sentiment_score is not None:
            if signals.sentiment_score > 0.5:
                score += 1.5
                reasons.append(f"Sentiment très positif ({signals.sentiment_score:+.2f})")
            elif signals.sentiment_score > 0.2:
                score += 0.75
                reasons.append(f"Sentiment positif ({signals.sentiment_score:+.2f})")
            elif signals.sentiment_score < -0.5:
                score -= 1.5
                reasons.append(f"Sentiment très négatif ({signals.sentiment_score:.2f})")
            elif signals.sentiment_score < -0.2:
                score -= 0.75
                reasons.append(f"Sentiment négatif ({signals.sentiment_score:.2f})")
        
        # --- Anomaly Detection ---
        if signals.has_anomaly:
            if signals.anomaly_severity and signals.anomaly_severity > 0.75:
                score -= 2.0
                reasons.append(
                    f"⚠️ Anomalie sévère détectée: {signals.anomaly_type}"
                )
            else:
                score -= 1.0
                reasons.append(f"Anomalie détectée: {signals.anomaly_type}")
        
        # --- Liquidity Assessment ---
        if signals.liquidity_tier:
            if self.risk_profile == RiskProfile.CONSERVATIVE:
                if signals.liquidity_tier == "low":
                    score -= 1.5
                    reasons.append("Liquidité faible (risque élevé)")
                elif signals.liquidity_tier == "high":
                    score += 0.5
                    reasons.append("Liquidité élevée")
            elif self.risk_profile == RiskProfile.AGGRESSIVE:
                # Aggressive traders may accept lower liquidity
                if signals.liquidity_tier == "low":
                    reasons.append("Liquidité faible (acceptable)")
        
        # --- Volume Analysis ---
        if signals.volume_change is not None:
            if signals.volume_change > 100:  # 2x normal volume
                score += 0.5
                reasons.append(f"Volume élevé (+{signals.volume_change:.0f}%)")
            elif signals.volume_change < -50:
                score -= 0.5
                reasons.append(f"Volume faible ({signals.volume_change:.0f}%)")
        
        # --- Convert score to signal ---
        signal, strength = self._score_to_signal(score)
        
        # --- Risk profile adjustments ---
        signal = self._adjust_for_risk_profile(signal, signals)
        
        logger.info(
            f"Rule evaluation for {signals.symbol}: "
            f"score={score:.2f}, signal={signal}, strength={strength}"
        )
        
        return signal, strength, reasons
    
    def _score_to_signal(self, score: float) -> tuple[Signal, SignalStrength]:
        """Convert numeric score to signal and strength."""
        if score >= 3.0:
            return Signal.STRONG_BUY, SignalStrength.VERY_HIGH
        elif score >= 1.5:
            return Signal.BUY, SignalStrength.HIGH
        elif score >= 0.5:
            return Signal.BUY, SignalStrength.MEDIUM
        elif score <= -3.0:
            return Signal.STRONG_SELL, SignalStrength.VERY_HIGH
        elif score <= -1.5:
            return Signal.SELL, SignalStrength.HIGH
        elif score <= -0.5:
            return Signal.SELL, SignalStrength.MEDIUM
        else:
            return Signal.HOLD, SignalStrength.LOW
    
    def _adjust_for_risk_profile(
        self,
        signal: Signal,
        signals: MarketSignals
    ) -> Signal:
        """
        Adjust signal based on risk profile.
        
        Conservative: Downgrade buy signals if any warnings
        Aggressive: Upgrade signals on high conviction
        """
        if self.risk_profile == RiskProfile.CONSERVATIVE:
            # Conservative: Be more cautious
            if signals.has_anomaly and signal in [Signal.BUY, Signal.STRONG_BUY]:
                # Downgrade buy signals when anomalies present
                if signal == Signal.STRONG_BUY:
                    signal = Signal.BUY
                else:
                    signal = Signal.HOLD
                logger.debug(f"Downgraded signal for conservative profile")
            
            # Require higher confidence
            if signals.confidence_score and signals.confidence_score < 0.75:
                if signal == Signal.STRONG_BUY:
                    signal = Signal.BUY
        
        elif self.risk_profile == RiskProfile.AGGRESSIVE:
            # Aggressive: More willing to take risks
            if signal == Signal.BUY and signals.predicted_return and signals.predicted_return > 7.0:
                signal = Signal.STRONG_BUY
                logger.debug(f"Upgraded to STRONG_BUY for aggressive profile")
        
        return signal
    
    def prioritize_recommendations(
        self,
        recommendations: List[tuple[str, Signal, SignalStrength, List[str]]]
    ) -> List[tuple[str, Signal, SignalStrength, List[str]]]:
        """
        Prioritize and rank recommendations.
        
        Args:
            recommendations: List of (symbol, signal, strength, reasons)
        
        Returns:
            Sorted list with best opportunities first
        """
        # Define priority scores
        signal_priority = {
            Signal.STRONG_BUY: 5,
            Signal.BUY: 4,
            Signal.HOLD: 2,
            Signal.SELL: 1,
            Signal.STRONG_SELL: 0
        }
        
        strength_priority = {
            SignalStrength.VERY_HIGH: 5,
            SignalStrength.HIGH: 4,
            SignalStrength.MEDIUM: 3,
            SignalStrength.LOW: 2,
            SignalStrength.VERY_LOW: 1
        }
        
        def priority_score(item):
            symbol, signal, strength, reasons = item
            return (
                signal_priority.get(signal, 0) * 10 +
                strength_priority.get(strength, 0)
            )
        
        sorted_recs = sorted(
            recommendations,
            key=priority_score,
            reverse=True
        )
        
        return sorted_recs
    
    def filter_by_profile(
        self,
        recommendations: List[tuple[str, Signal, SignalStrength, List[str]]],
        max_count: int = 10
    ) -> List[tuple[str, Signal, SignalStrength, List[str]]]:
        """
        Filter recommendations based on risk profile.
        
        Args:
            recommendations: List of recommendations
            max_count: Maximum recommendations to return
        
        Returns:
            Filtered and limited list
        """
        filtered = []
        
        for symbol, signal, strength, reasons in recommendations:
            # Conservative: Only show high-confidence buys
            if self.risk_profile == RiskProfile.CONSERVATIVE:
                if signal in [Signal.STRONG_BUY, Signal.BUY]:
                    if strength in [SignalStrength.VERY_HIGH, SignalStrength.HIGH]:
                        filtered.append((symbol, signal, strength, reasons))
            
            # Moderate: Show most signals
            elif self.risk_profile == RiskProfile.MODERATE:
                if signal != Signal.HOLD:
                    filtered.append((symbol, signal, strength, reasons))
            
            # Aggressive: Show all signals
            else:
                filtered.append((symbol, signal, strength, reasons))
            
            if len(filtered) >= max_count:
                break
        
        return filtered
