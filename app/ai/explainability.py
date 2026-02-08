"""
Explainability Module with Groq AI Integration.

Generates natural language explanations for trading recommendations
using Groq's fast LLM inference API.

Each recommendation is explained with:
- Why the recommendation was made
- Supporting evidence from multiple signals
- Risk considerations
- Alternative perspectives

This module is critical for user trust and regulatory compliance.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("groq package not installed. Install with: pip install groq")

from app.ai.config import ai_settings
from app.ai.rules import Signal, SignalStrength, MarketSignals

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generates natural language explanations for trading decisions.
    
    Uses Groq API for fast, coherent explanations that combine
    multiple data sources into understandable narratives.
    """
    
    def __init__(self):
        """Initialize Groq client."""
        if not GROQ_AVAILABLE:
            logger.error("Groq package not available. Install with: pip install groq")
            self.client = None
            return
        
        if not ai_settings.groq_api_key:
            logger.warning("GROQ_API_KEY not configured. Explanations will be rule-based only.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=ai_settings.groq_api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
    
    async def explain_recommendation(
        self,
        symbol: str,
        signal: Signal,
        strength: SignalStrength,
        signals: MarketSignals,
        reasons: List[str],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate detailed explanation for a recommendation.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal (BUY/SELL/HOLD)
            strength: Signal strength
            signals: All market signals
            reasons: Rule-based reasons
            user_context: Additional user context (portfolio, preferences)
        
        Returns:
            Natural language explanation
        """
        # If Groq not available, use fallback
        if self.client is None:
            return self._generate_fallback_explanation(
                symbol, signal, strength, signals, reasons
            )
        
        try:
            # Build context for LLM
            context = self._build_context(
                symbol, signal, strength, signals, reasons, user_context
            )
            
            # Generate explanation using Groq
            explanation = await self._query_groq(context)
            
            logger.info(f"Generated explanation for {symbol} via Groq")
            return explanation
            
        except Exception as e:
            logger.error(f"Groq API error: {e}. Falling back to rule-based explanation.")
            return self._generate_fallback_explanation(
                symbol, signal, strength, signals, reasons
            )
    
    def _build_context(
        self,
        symbol: str,
        signal: Signal,
        strength: SignalStrength,
        signals: MarketSignals,
        reasons: List[str],
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string for LLM."""
        
        # Prepare structured data
        data = {
            "symbol": symbol,
            "recommendation": signal.value,
            "strength": strength.value,
            "price_prediction": {
                "predicted_return": signals.predicted_return,
                "confidence": signals.confidence_score
            },
            "sentiment": {
                "score": signals.sentiment_score,
                "label": signals.sentiment_label
            },
            "anomaly": {
                "detected": signals.has_anomaly,
                "type": signals.anomaly_type,
                "severity": signals.anomaly_severity
            },
            "liquidity": {
                "tier": signals.liquidity_tier,
                "probability": signals.liquidity_prob
            },
            "volume": {
                "predicted": signals.predicted_volume,
                "change_pct": signals.volume_change
            },
            "rule_based_reasons": reasons
        }
        
        if user_context:
            data["user_context"] = user_context
        
        context = f"""Tu es un expert en analyse financière de la Bourse de Valeurs Mobilières de Tunis (BVMT).

Génère une explication claire et concise pour une recommandation de trading basée sur les données suivantes :

{json.dumps(data, indent=2, ensure_ascii=False)}

L'explication doit :
1. Commencer par la recommandation principale (ACHAT/VENTE/CONSERVATION)
2. Expliquer les 3 raisons principales en langage naturel
3. Mentionner les risques ou points d'attention
4. Être compréhensible pour un investisseur non-expert
5. Utiliser des termes en français
6. Faire maximum 150 mots

Ne répète pas les chiffres bruts, mais intègre-les naturellement dans l'explication.
"""
        
        return context
    
    async def _query_groq(self, context: str) -> str:
        """Query Groq API for explanation."""
        try:
            completion = self.client.chat.completions.create(
                model=ai_settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un conseiller financier expert en bourse tunisienne."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=ai_settings.groq_temperature,
                max_tokens=ai_settings.groq_max_tokens,
                top_p=0.9,
                stream=False
            )
            
            explanation = completion.choices[0].message.content
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
    
    def _generate_fallback_explanation(
        self,
        symbol: str,
        signal: Signal,
        strength: SignalStrength,
        signals: MarketSignals,
        reasons: List[str]
    ) -> str:
        """
        Generate rule-based explanation when Groq is unavailable.
        
        This is a fallback that provides basic explanations without LLM.
        """
        # Signal translation
        signal_text = {
            Signal.STRONG_BUY: "**ACHAT FORT**",
            Signal.BUY: "**ACHAT**",
            Signal.HOLD: "**CONSERVER**",
            Signal.SELL: "**VENTE**",
            Signal.STRONG_SELL: "**VENTE FORTE**"
        }
        
        strength_text = {
            SignalStrength.VERY_HIGH: "Conviction très élevée",
            SignalStrength.HIGH: "Conviction élevée",
            SignalStrength.MEDIUM: "Conviction moyenne",
            SignalStrength.LOW: "Conviction faible",
            SignalStrength.VERY_LOW: "Conviction très faible"
        }
        
        parts = [
            f"**Recommandation pour {symbol} : {signal_text[signal]}**",
            f"\n{strength_text[strength]}",
            "\n\n**Analyse :**"
        ]
        
        # Add reasons
        if reasons:
            parts.append("\n• " + "\n• ".join(reasons[:5]))  # Top 5 reasons
        
        # Add prediction summary
        if signals.predicted_return is not None:
            parts.append(
                f"\n\n**Prévision :** {signals.predicted_return:+.1f}% "
                f"(confiance: {signals.confidence_score:.0%})"
            )
        
        # Add sentiment
        if signals.sentiment_score is not None:
            sentiment_emoji = "📈" if signals.sentiment_score > 0 else "📉"
            parts.append(
                f"\n**Sentiment :** {sentiment_emoji} {signals.sentiment_label}"
            )
        
        # Add warnings
        warnings = []
        if signals.has_anomaly:
            warnings.append(f"⚠️ Anomalie détectée : {signals.anomaly_type}")
        
        if signals.confidence_score and signals.confidence_score < 0.65:
            warnings.append("⚠️ Confiance de prévision faible")
        
        if signals.liquidity_tier == "low":
            warnings.append("⚠️ Liquidité faible")
        
        if warnings:
            parts.append("\n\n**Points d'attention :**")
            parts.append("\n• " + "\n• ".join(warnings))
        
        explanation = "".join(parts)
        
        logger.info(f"Generated fallback explanation for {symbol}")
        return explanation
    
    async def explain_portfolio_action(
        self,
        action: str,  # "buy" or "sell"
        symbol: str,
        quantity: int,
        price: float,
        portfolio_context: Dict[str, Any]
    ) -> str:
        """
        Explain why a portfolio action was taken.
        
        Args:
            action: "buy" or "sell"
            symbol: Stock symbol
            quantity: Number of shares
            price: Execution price
            portfolio_context: Current portfolio state
        
        Returns:
            Explanation of the action
        """
        action_text = "Achat" if action == "buy" else "Vente"
        
        context = f"""Explique pourquoi l'action suivante a été effectuée :

Action : {action_text} de {quantity} actions {symbol} à {price:.3f} TND

Contexte du portefeuille :
- Capital total : {portfolio_context.get('total_value', 0):.2f} TND
- Liquidités : {portfolio_context.get('cash_balance', 0):.2f} TND
- Allocation actions : {portfolio_context.get('equity_allocation', 0):.1%}
- Positions actuelles : {portfolio_context.get('position_count', 0)}

Génère une explication courte (2-3 phrases) en français.
"""
        
        if self.client is None:
            return f"{action_text} de {quantity} {symbol} à {price:.3f} TND effectué."
        
        try:
            explanation = await self._query_groq(context)
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate portfolio action explanation: {e}")
            return f"{action_text} de {quantity} {symbol} à {price:.3f} TND effectué."
    
    async def explain_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Explain portfolio metrics in simple terms.
        
        Args:
            metrics: Portfolio metrics dictionary
        
        Returns:
            Natural language explanation
        """
        if self.client is None:
            return self._format_metrics_fallback(metrics)
        
        context = f"""Explique les performances d'un portefeuille d'investissement :

Métriques :
- ROI : {metrics.get('roi', 0):.2f}%
- Sharpe Ratio : {metrics.get('sharpe_ratio', 0):.2f}
- Drawdown Maximum : {metrics.get('max_drawdown', 0):.2f}%
- Volatilité : {metrics.get('volatility', 0):.2f}%
- Taux de réussite : {metrics.get('win_rate', 0):.1f}%

Génère une interprétation claire et actionnable en 3-4 phrases, en français.
Utilise des termes simples pour un investisseur non-expert.
"""
        
        try:
            explanation = await self._query_groq(context)
            return explanation
        except Exception as e:
            logger.error(f"Failed to explain metrics: {e}")
            return self._format_metrics_fallback(metrics)
    
    def _format_metrics_fallback(self, metrics: Dict[str, Any]) -> str:
        """Fallback formatting for metrics."""
        roi = metrics.get('roi', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        performance = "positive" if roi > 0 else "négative"
        quality = "excellente" if sharpe > 2 else "bonne" if sharpe > 1 else "moyenne"
        
        return (
            f"Performance {performance} avec un ROI de {roi:.2f}%. "
            f"La qualité risque/rendement est {quality} (Sharpe: {sharpe:.2f}). "
            f"Le drawdown maximum est de {metrics.get('max_drawdown', 0):.2f}%."
        )
