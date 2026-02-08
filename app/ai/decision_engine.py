"""
Decision Agent for Portfolio Management.

Generates BUY/SELL/HOLD recommendations based on:
- Portfolio optimization
- CAPM expected returns
- Risk contribution analysis
- User profile constraints
- Anomaly penalties

Every decision is explainable and deterministic.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from app.ai.optimization import PortfolioOptimizer, CAPMCalculator, PortfolioWeights
from app.ai.profile import RiskProfile
from app.ai.llm_explainer import LLMExplainer, ExplanationContext, LLMConfig

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    """Trading decision."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class AssetSignal:
    """Complete signal for an asset."""
    symbol: str
    decision: Decision
    confidence: float
    optimal_weight: float
    current_weight: float
    weight_delta: float
    expected_return: float
    capm_return: float
    contribution_to_risk: float
    beta: float
    anomaly_detected: bool
    diversification_benefit: float
    reasons: List[str]


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    timestamp: datetime
    risk_profile: RiskProfile
    total_value: float
    cash_available: float
    signals: List[AssetSignal]
    optimal_portfolio: PortfolioWeights
    rebalancing_needed: bool
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class DecisionEngine:
    """
    Decision engine for portfolio management.
    
    Generates actionable BUY/SELL/HOLD recommendations.
    """
    
    def __init__(
        self,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        risk_free_rate: float = 0.05,
        market_premium: float = 0.08,
        anomaly_penalty: float = 0.03,
        rebalancing_threshold: float = 0.05,
        use_llm_explanation: bool = False,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            risk_profile: User risk profile
            risk_free_rate: Annual risk-free rate
            market_premium: Market risk premium
            anomaly_penalty: Return penalty for anomalies
            rebalancing_threshold: Minimum weight delta for rebalancing
            use_llm_explanation: Enable LLM explanations
            llm_config: LLM configuration from frontend
        """
        self.risk_profile = risk_profile
        self.optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
        self.capm = CAPMCalculator(
            risk_free_rate=risk_free_rate,
            market_premium=market_premium
        )
        self.anomaly_penalty = anomaly_penalty
        self.rebalancing_threshold = rebalancing_threshold
        self.explainer = LLMExplainer(
            use_llm=use_llm_explanation,
            llm_config=llm_config
        )
    
    def calculate_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate marginal risk contribution of each asset.
        
        MRC_i = (w^T * Cov * e_i) / sigma_p
        
        Where e_i is unit vector for asset i.
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
        
        Returns:
            Array of risk contributions (sum to 1)
        """
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return np.zeros(len(weights))
        
        marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_contributions
        
        total_risk = np.sum(risk_contributions)
        if total_risk > 0:
            risk_contributions = risk_contributions / total_risk
        
        return risk_contributions
    
    def adjust_returns_for_anomalies(
        self,
        expected_returns: Dict[str, float],
        anomalies: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Penalize expected returns for assets with anomalies.
        
        Args:
            expected_returns: Dict of symbol -> expected return
            anomalies: Dict of symbol -> has_anomaly
        
        Returns:
            Adjusted expected returns
        """
        adjusted = expected_returns.copy()
        
        for symbol, has_anomaly in anomalies.items():
            if has_anomaly and symbol in adjusted:
                adjusted[symbol] -= self.anomaly_penalty
                logger.info(
                    f"Applied anomaly penalty to {symbol}: "
                    f"{expected_returns[symbol]:.2%} -> {adjusted[symbol]:.2%}"
                )
        
        return adjusted
    
    def calculate_diversification_benefit(
        self,
        asset_returns: pd.Series,
        portfolio_returns: pd.Series
    ) -> float:
        """
        Calculate how much an asset improves diversification.
        
        Benefit = 1 - |correlation(asset, portfolio)|
        
        Higher is better (lower correlation = more diversification).
        
        Args:
            asset_returns: Asset return series
            portfolio_returns: Current portfolio return series
        
        Returns:
            Diversification benefit (0 to 1)
        """
        correlation = asset_returns.corr(portfolio_returns)
        benefit = 1.0 - abs(correlation)
        return benefit
    
    def generate_decision(
        self,
        symbol: str,
        optimal_weight: float,
        current_weight: float,
        expected_return: float,
        capm_return: float,
        risk_contribution: float,
        beta: float,
        anomaly_detected: bool,
        diversification_benefit: float,
        risk_profile: RiskProfile
    ) -> Tuple[Decision, float, List[str]]:
        """
        Generate trading decision for an asset.
        
        Args:
            symbol: Asset symbol
            optimal_weight: Target weight from optimization
            current_weight: Current portfolio weight
            expected_return: Predicted return
            capm_return: CAPM expected return
            risk_contribution: Marginal risk contribution
            beta: CAPM beta
            anomaly_detected: Anomaly flag
            diversification_benefit: Diversification score
            risk_profile: User risk profile
        
        Returns:
            (decision, confidence, reasons)
        """
        weight_delta = optimal_weight - current_weight
        reasons = []
        
        if anomaly_detected:
            reasons.append(
                "Anomaly detected - increased risk exposure"
            )
        
        excess_return = expected_return - capm_return
        
        if abs(weight_delta) < self.rebalancing_threshold:
            decision = Decision.HOLD
            confidence = 0.7
            reasons.append(
                f"Current allocation ({current_weight:.1%}) is close to optimal ({optimal_weight:.1%})"
            )
        
        elif weight_delta > self.rebalancing_threshold:
            if anomaly_detected:
                decision = Decision.HOLD
                confidence = 0.5
                reasons.append(
                    "BUY signal overridden due to anomaly detection"
                )
            elif excess_return > 0:
                if weight_delta > 0.10:
                    decision = Decision.STRONG_BUY
                    confidence = 0.9
                    reasons.append(
                        f"Strong positive signal: expected return ({expected_return:.2%}) "
                        f"exceeds CAPM benchmark ({capm_return:.2%})"
                    )
                else:
                    decision = Decision.BUY
                    confidence = 0.8
                    reasons.append(
                        f"Positive expected return ({expected_return:.2%}) above CAPM ({capm_return:.2%})"
                    )
                
                reasons.append(
                    f"Optimal weight ({optimal_weight:.1%}) > current ({current_weight:.1%})"
                )
                
                if diversification_benefit > 0.5:
                    reasons.append(
                        f"Strong diversification benefit (score: {diversification_benefit:.2f})"
                    )
            else:
                decision = Decision.HOLD
                confidence = 0.6
                reasons.append(
                    "Underweighted but expected return below CAPM benchmark"
                )
        
        elif weight_delta < -self.rebalancing_threshold:
            if excess_return < 0:
                if weight_delta < -0.10:
                    decision = Decision.STRONG_SELL
                    confidence = 0.9
                    reasons.append(
                        f"Strong negative signal: expected return ({expected_return:.2%}) "
                        f"below CAPM benchmark ({capm_return:.2%})"
                    )
                else:
                    decision = Decision.SELL
                    confidence = 0.8
                    reasons.append(
                        f"Expected return ({expected_return:.2%}) below CAPM ({capm_return:.2%})"
                    )
                
                reasons.append(
                    f"Optimal weight ({optimal_weight:.1%}) < current ({current_weight:.1%})"
                )
            else:
                decision = Decision.SELL
                confidence = 0.7
                reasons.append(
                    "Rebalancing required despite positive expected return"
                )
                reasons.append(
                    f"Reduces portfolio risk contribution ({risk_contribution:.2%})"
                )
        
        else:
            decision = Decision.HOLD
            confidence = 0.8
            reasons.append(
                f"Current allocation is optimal for {risk_profile.value} profile"
            )
        
        if risk_contribution > 0.20:
            reasons.append(
                f"High risk contribution ({risk_contribution:.1%}) to portfolio"
            )
        
        if beta > 1.5:
            reasons.append(
                f"High systematic risk (beta: {beta:.2f})"
            )
        elif beta < 0.7:
            reasons.append(
                f"Defensive asset (beta: {beta:.2f})"
            )
        
        if risk_profile == RiskProfile.CONSERVATIVE:
            if decision in [Decision.STRONG_BUY, Decision.BUY] and beta > 1.2:
                decision = Decision.HOLD
                confidence *= 0.8
                reasons.append(
                    "BUY signal downgraded for conservative profile due to high beta"
                )
        
        elif risk_profile == RiskProfile.AGGRESSIVE:
            if decision in [Decision.STRONG_SELL, Decision.SELL] and excess_return > 0:
                decision = Decision.HOLD
                confidence *= 0.9
                reasons.append(
                    "SELL signal moderated for aggressive profile due to positive alpha"
                )
        
        return decision, confidence, reasons
    
    def generate_recommendations(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        current_portfolio: Dict[str, float],
        risk_profile: RiskProfile,
        predictions: Optional[Dict[str, float]] = None,
        anomalies: Optional[Dict[str, bool]] = None,
        total_value: float = 10000.0,
        cash_available: float = 0.0
    ) -> PortfolioRecommendation:
        """
        Generate complete portfolio recommendations.
        
        Args:
            returns: DataFrame of historical returns
            market_returns: Market return series
            current_portfolio: Dict of symbol -> current weight
            risk_profile: User risk profile
            predictions: Optional predicted returns
            anomalies: Optional anomaly flags
            total_value: Total portfolio value
            cash_available: Available cash
        
        Returns:
            PortfolioRecommendation with all signals
        """
        symbols = returns.columns.tolist()
        
        capm_returns = self.capm.calculate_all_expected_returns(returns, market_returns)
        betas = self.capm.calculate_all_betas(returns, market_returns)
        
        if predictions:
            expected_returns = predictions
        else:
            expected_returns = capm_returns
        
        if anomalies:
            expected_returns = self.adjust_returns_for_anomalies(
                expected_returns,
                anomalies
            )
        else:
            anomalies = dict.fromkeys(symbols, False)
        
        optimal_portfolio = self.optimizer.optimize_for_profile(
            returns,
            risk_profile,
            expected_returns
        )
        
        cov_matrix = self.optimizer.calculate_covariance_matrix(returns)
        risk_contributions = self.calculate_risk_contribution(
            optimal_portfolio.weights,
            cov_matrix
        )
        
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for sym, weight in current_portfolio.items():
            if sym in returns.columns:
                portfolio_returns += returns[sym] * weight
        
        signals = []
        
        for i, symbol in enumerate(symbols):
            current_weight = current_portfolio.get(symbol, 0.0)
            optimal_weight = optimal_portfolio.weights[i]
            
            diversification_benefit = self.calculate_diversification_benefit(
                returns[symbol],
                portfolio_returns
            ) if len(current_portfolio) > 0 else 0.8
            
            decision, confidence, reasons = self.generate_decision(
                symbol=symbol,
                optimal_weight=optimal_weight,
                current_weight=current_weight,
                expected_return=expected_returns.get(symbol, 0.0),
                capm_return=capm_returns[symbol],
                risk_contribution=risk_contributions[i],
                beta=betas[symbol],
                anomaly_detected=anomalies.get(symbol, False),
                diversification_benefit=diversification_benefit,
                risk_profile=risk_profile
            )
            
            signal = AssetSignal(
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                optimal_weight=optimal_weight,
                current_weight=current_weight,
                weight_delta=optimal_weight - current_weight,
                expected_return=expected_returns.get(symbol, 0.0),
                capm_return=capm_returns[symbol],
                contribution_to_risk=risk_contributions[i],
                beta=betas[symbol],
                anomaly_detected=anomalies.get(symbol, False),
                diversification_benefit=diversification_benefit,
                reasons=reasons
            )
            
            signals.append(signal)
        
        signals.sort(key=lambda s: abs(s.weight_delta), reverse=True)
        
        rebalancing_needed = any(
            abs(s.weight_delta) > self.rebalancing_threshold
            for s in signals
        )
        
        recommendation = PortfolioRecommendation(
            timestamp=datetime.now(),
            risk_profile=risk_profile,
            total_value=total_value,
            cash_available=cash_available,
            signals=signals,
            optimal_portfolio=optimal_portfolio,
            rebalancing_needed=rebalancing_needed,
            expected_return=optimal_portfolio.expected_return,
            expected_volatility=optimal_portfolio.expected_volatility,
            sharpe_ratio=optimal_portfolio.sharpe_ratio
        )
        
        logger.info(
            f"Generated recommendations for {risk_profile.value}: "
            f"{len([s for s in signals if s.decision in [Decision.BUY, Decision.STRONG_BUY]])} BUY, "
            f"{len([s for s in signals if s.decision in [Decision.SELL, Decision.STRONG_SELL]])} SELL, "
            f"{len([s for s in signals if s.decision == Decision.HOLD])} HOLD"
        )
        
        return recommendation
