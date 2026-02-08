"""
Prompt loader for LLM explainer.

Loads Wall Street-grade prompts from YAML configuration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class PromptLoader:
    """Load and manage system prompts from YAML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize prompt loader.
        
        Args:
            config_path: Path to prompts.yaml file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "prompts.yaml"
        
        self.config_path = config_path
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            logger.info(f"Loaded prompts from {self.config_path}")
            return prompts
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> Dict[str, Any]:
        """Fallback prompts if YAML fails to load."""
        return {
            "buy_signal": {
                "system": "You are a professional portfolio manager. Be concise and data-driven.",
                "user_template": "Explain this BUY: {symbol}, Return: {expected_return:.1f}%, Beta: {beta:.2f}"
            },
            "sell_signal": {
                "system": "You are a professional portfolio manager. Be concise and data-driven.",
                "user_template": "Explain this SELL: {symbol}, Return: {expected_return:.1f}%, Beta: {beta:.2f}"
            },
            "hold_signal": {
                "system": "You are a professional portfolio manager. Be concise and data-driven.",
                "user_template": "Explain this HOLD: {symbol}, Return: {expected_return:.1f}%, Beta: {beta:.2f}"
            }
        }
    
    def get_system_prompt(self, signal_type: str) -> str:
        """
        Get system prompt for signal type.
        
        Args:
            signal_type: 'buy_signal', 'sell_signal', 'hold_signal', or 'portfolio_summary'
        
        Returns:
            System prompt text
        """
        return self.prompts.get(signal_type, {}).get("system", "You are a portfolio manager.")
    
    def get_user_prompt(
        self,
        signal_type: str,
        symbol: str,
        expected_return: float,
        beta: float,
        current_weight: float,
        target_weight: float,
        risk_contribution: float,
        risk_profile: str,
        anomaly_detected: bool
    ) -> str:
        """
        Generate user prompt for asset-level decision.
        
        Args:
            signal_type: 'buy_signal', 'sell_signal', or 'hold_signal'
            symbol: Stock symbol
            expected_return: Expected return %
            beta: Beta coefficient
            current_weight: Current portfolio weight
            target_weight: Target portfolio weight
            risk_contribution: Risk contribution %
            risk_profile: Risk profile string
            anomaly_detected: Whether anomaly detected
        
        Returns:
            Formatted user prompt
        """
        template = self.prompts.get(signal_type, {}).get("user_template", "")
        
        if not template:
            return f"Explain {signal_type} for {symbol}"
        
        anomaly_status = "YES - PROCEED WITH CAUTION" if anomaly_detected else "NO"
        
        try:
            return template.format(
                symbol=symbol,
                expected_return=expected_return,
                beta=beta,
                current_weight=current_weight,
                target_weight=target_weight,
                risk_contribution=risk_contribution,
                risk_profile=risk_profile.upper(),
                anomaly_status=anomaly_status
            )
        except Exception as e:
            logger.error(f"Failed to format prompt: {e}")
            return f"Explain {signal_type} for {symbol}: {expected_return:.1f}% return, {beta:.2f} beta"
    
    def get_portfolio_prompt(
        self,
        risk_profile: str,
        expected_return: float,
        volatility: float,
        sharpe_ratio: float,
        diversification_ratio: float,
        buy_count: int,
        sell_count: int,
        hold_count: int
    ) -> str:
        """
        Generate portfolio summary prompt.
        
        Args:
            risk_profile: Risk profile
            expected_return: Portfolio expected return %
            volatility: Portfolio volatility %
            sharpe_ratio: Sharpe ratio
            diversification_ratio: Diversification ratio
            buy_count: Number of BUY signals
            sell_count: Number of SELL signals
            hold_count: Number of HOLD signals
        
        Returns:
            Formatted portfolio prompt
        """
        template = self.prompts.get("portfolio_summary", {}).get("user_template", "")
        
        if not template:
            return f"Summarize portfolio: {expected_return:.1f}% return, {sharpe_ratio:.2f} Sharpe"
        
        try:
            return template.format(
                risk_profile=risk_profile.upper(),
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio,
                buy_count=buy_count,
                sell_count=sell_count,
                hold_count=hold_count
            )
        except Exception as e:
            logger.error(f"Failed to format portfolio prompt: {e}")
            return f"Summarize portfolio strategy"
    
    def get_confidence_modifier(self, confidence: float) -> str:
        """
        Get confidence modifier suffix.
        
        Args:
            confidence: Confidence percentage (0-100)
        
        Returns:
            Confidence modifier text
        """
        import random
        
        if confidence >= 80:
            modifiers = self.prompts.get("high_confidence_suffix", [])
        elif confidence >= 60:
            modifiers = self.prompts.get("moderate_confidence_suffix", [])
        else:
            modifiers = self.prompts.get("low_confidence_suffix", [])
        
        return random.choice(modifiers) if modifiers else ""


# Global prompt loader instance
_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get global prompt loader instance."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader
