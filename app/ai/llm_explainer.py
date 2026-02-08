"""
LLM-based explainability layer for portfolio recommendations.

Uses LiteLLM for provider-agnostic LLM access with template-based fallback.
Loads Wall Street-grade prompts from YAML configuration.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.ai.prompt_loader import get_prompt_loader

logger = logging.getLogger(__name__)


@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    symbol: str
    action: str
    confidence: float
    current_weight: float
    target_weight: float
    expected_return: float
    beta: float
    risk_contribution: float
    anomaly_detected: bool
    sentiment_score: Optional[float] = None
    risk_profile: str = "moderate"


@dataclass
class LLMConfig:
    """LLM configuration from frontend."""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 150
    enable_reasoning: bool = False


class LLMExplainer:
    """
    Generate natural language explanations for trading decisions.
    
    Supports multiple LLM providers:
    - OpenRouter (default, with reasoning support)
    - OpenAI
    - Anthropic
    - Groq
    
    Falls back to template-based generation if LLM unavailable.
    API key and model selection controlled by frontend.
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize explainer.
        
        Args:
            use_llm: Enable LLM-based explanations
            llm_config: LLM configuration (provider, model, API key from frontend)
        """
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.use_openrouter = False
        self.prompt_loader = get_prompt_loader()
        
        if self.use_llm and llm_config:
            # Check if using OpenRouter
            if llm_config.provider.lower() == "openrouter":
                self.use_openrouter = True
                logger.info(
                    f"OpenRouter initialized: model={llm_config.model}, "
                    f"reasoning={llm_config.enable_reasoning}"
                )
            else:
                # Try to use LiteLLM for other providers
                try:
                    import litellm
                    self.litellm = litellm
                    
                    # Set API key for the provider
                    import os
                    if llm_config.provider.lower() == "openai":
                        os.environ["OPENAI_API_KEY"] = llm_config.api_key
                    elif llm_config.provider.lower() == "anthropic":
                        os.environ["ANTHROPIC_API_KEY"] = llm_config.api_key
                    elif llm_config.provider.lower() == "groq":
                        os.environ["GROQ_API_KEY"] = llm_config.api_key
                    else:
                        os.environ[f"{llm_config.provider.upper()}_API_KEY"] = llm_config.api_key
                    
                    logger.info(
                        f"LiteLLM initialized: provider={llm_config.provider}, "
                        f"model={llm_config.model}"
                    )
                except ImportError:
                    logger.warning("LiteLLM not installed, falling back to templates")
                    self.use_llm = False
                except Exception as e:
                    logger.error(f"LiteLLM initialization failed: {e}")
                    self.use_llm = False
    
    def explain_decision(
        self,
        context: ExplanationContext
    ) -> str:
        """
        Generate explanation for a trading decision.
        
        Args:
            context: Decision context
        
        Returns:
            Natural language explanation
        """
        if self.use_llm:
            try:
                return self._generate_llm_explanation(context)
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
        
        return self._generate_template_explanation(context)
    
    def _generate_llm_explanation(
        self,
        context: ExplanationContext
    ) -> str:
        """Generate explanation using LLM."""
        if not self.llm_config:
            raise ValueError("LLM config not provided")
        
        system_prompt, user_prompt = self._build_prompt(context)
        
        try:
            if self.use_openrouter:
                return self._openrouter_completion(system_prompt, user_prompt)
            else:
                return self._litellm_completion(system_prompt, user_prompt)
        
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise
    
    def _openrouter_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate explanation using OpenRouter API."""
        import requests
        import json
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        payload = {
            "model": self.llm_config.model,
            "messages": messages,
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens
        }
        
        # Add reasoning if enabled
        if self.llm_config.enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        explanation = result['choices'][0]['message']['content'].strip()
        
        # Log if reasoning was used
        if self.llm_config.enable_reasoning:
            reasoning_details = result['choices'][0]['message'].get('reasoning_details')
            if reasoning_details:
                logger.debug(f"Reasoning tokens used: {reasoning_details}")
        
        return explanation
    
    def _litellm_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate explanation using LiteLLM."""
        response = self.litellm.completion(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _build_prompt(
        self,
        context: ExplanationContext
    ) -> tuple:
        """
        Build LLM prompt from context using YAML prompts.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Determine signal type
        signal_type = f"{context.action.lower()}_signal"
        
        # Get system prompt
        system_prompt = self.prompt_loader.get_system_prompt(signal_type)
        
        # Get user prompt
        user_prompt = self.prompt_loader.get_user_prompt(
            signal_type=signal_type,
            symbol=context.symbol,
            expected_return=context.expected_return,
            beta=context.beta,
            current_weight=context.current_weight * 100,  # Convert to %
            target_weight=context.target_weight * 100,
            risk_contribution=context.risk_contribution,
            risk_profile=context.risk_profile,
            anomaly_detected=context.anomaly_detected
        )
        
        return system_prompt, user_prompt
    
    def _generate_template_explanation(
        self,
        context: ExplanationContext
    ) -> str:
        """Generate explanation using templates."""
        
        if context.action == "BUY":
            return self._explain_buy(context)
        elif context.action == "SELL":
            return self._explain_sell(context)
        elif context.action == "HOLD":
            return self._explain_hold(context)
        else:
            return f"{context.action} signal generated with {context.confidence:.1f}% confidence."
    
    def _explain_buy(self, context: ExplanationContext) -> str:
        """Template for BUY decisions."""
        parts = []
        
        if context.expected_return > 8:
            parts.append(
                f"Expected return ({context.expected_return:.1f}%) "
                "exceeds CAPM benchmark"
            )
        else:
            parts.append(
                f"Expected return ({context.expected_return:.1f}%) "
                "aligns with risk-adjusted expectations"
            )
        
        if context.beta < 1.0:
            parts.append("with below-market systematic risk")
        elif context.beta > 1.2:
            parts.append("accepting elevated market sensitivity")
        
        if context.current_weight < context.target_weight:
            weight_gap = context.target_weight - context.current_weight
            parts.append(
                f"to increase position by {weight_gap:.1f}% "
                f"for {context.risk_profile} portfolio optimization"
            )
        
        if context.anomaly_detected:
            parts.append("(anomaly detected - proceed with caution)")
        
        return " ".join(parts) + "."
    
    def _explain_sell(self, context: ExplanationContext) -> str:
        """Template for SELL decisions."""
        parts = []
        
        if context.expected_return < 3:
            parts.append(
                f"Expected return ({context.expected_return:.1f}%) "
                "below acceptable threshold"
            )
        else:
            parts.append("Portfolio rebalancing required")
        
        if context.risk_contribution > 30:
            parts.append(
                f"to reduce excessive risk contribution "
                f"({context.risk_contribution:.1f}%)"
            )
        elif context.current_weight > context.target_weight:
            weight_excess = context.current_weight - context.target_weight
            parts.append(
                f"to decrease overweight position by {weight_excess:.1f}%"
            )
        
        if context.anomaly_detected:
            parts.append("Anomaly detected - risk mitigation warranted")
        
        return " ".join(parts) + "."
    
    def _explain_hold(self, context: ExplanationContext) -> str:
        """Template for HOLD decisions."""
        parts = []
        
        weight_delta = abs(context.target_weight - context.current_weight)
        
        if weight_delta < 1:
            parts.append("Position already optimally weighted")
        else:
            parts.append("Current allocation acceptable")
        
        parts.append(
            f"Expected return ({context.expected_return:.1f}%) "
            f"and risk contribution ({context.risk_contribution:.1f}%) "
            "within target ranges"
        )
        
        if context.beta < 0.8:
            parts.append("Low beta provides defensive characteristics")
        elif context.beta > 1.2:
            parts.append("High beta monitored for volatility")
        
        return " ".join(parts) + "."
    
    def explain_portfolio(
        self,
        recommendations: List[Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        risk_profile: str
    ) -> str:
        """
        Generate portfolio-level explanation.
        
        Args:
            recommendations: List of asset recommendations
            portfolio_metrics: Portfolio-level metrics
            risk_profile: User risk profile
        
        Returns:
            Portfolio-level explanation
        """
        if self.use_llm:
            try:
                return self._generate_portfolio_llm_explanation(
                    recommendations,
                    portfolio_metrics,
                    risk_profile
                )
            except Exception as e:
                logger.error(f"LLM portfolio explanation failed: {e}")
        
        return self._generate_portfolio_template_explanation(
            recommendations,
            portfolio_metrics,
            risk_profile
        )
    
    def _generate_portfolio_llm_explanation(
        self,
        recommendations: List[Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        risk_profile: str
    ) -> str:
        """Generate portfolio explanation using LLM."""
        if not self.llm_config:
            raise ValueError("LLM config not provided")
        
        buy_count = sum(1 for r in recommendations if r.get("action") == "BUY")
        sell_count = sum(1 for r in recommendations if r.get("action") == "SELL")
        hold_count = sum(1 for r in recommendations if r.get("action") == "HOLD")
        
        expected_return = portfolio_metrics.get('expected_return', 0)
        volatility = portfolio_metrics.get('volatility', 0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        diversification_ratio = portfolio_metrics.get('diversification_ratio', 0)
        
        try:
            if self.use_openrouter:
                return self._openrouter_portfolio_completion(
                    risk_profile, expected_return, volatility,
                    sharpe_ratio, diversification_ratio,
                    buy_count, sell_count, hold_count
                )
            else:
                return self._litellm_portfolio_completion(
                    risk_profile, expected_return, volatility,
                    sharpe_ratio, diversification_ratio,
                    buy_count, sell_count, hold_count
                )
        
        except Exception as e:
            logger.error(f"LLM portfolio completion failed: {e}")
            raise
    
    def _openrouter_portfolio_completion(self, prompt: str) -> str:
        """Generate portfolio explanation using OpenRouter."""
        import requests
        import json
        
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a portfolio manager summarizing strategy. "
                        "Be concise and highlight key risk-return tradeoffs."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.llm_config.temperature,
            "max_tokens": 200
        }
        
        if self.llm_config.enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content'].strip()
    
    def _litellm_portfolio_completion(
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
        """Generate portfolio explanation using LiteLLM."""
        system_prompt = self.prompt_loader.get_system_prompt("portfolio_summary")
        user_prompt = self.prompt_loader.get_portfolio_prompt(
            risk_profile=risk_profile,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count
        )
        
        response = self.litellm.completion(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=self.llm_config.temperature,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_portfolio_template_explanation(
        self,
        recommendations: List[Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        risk_profile: str
    ) -> str:
        """Generate portfolio explanation using templates."""
        
        buy_count = sum(1 for r in recommendations if r.get("action") == "BUY")
        sell_count = sum(1 for r in recommendations if r.get("action") == "SELL")
        
        expected_return = portfolio_metrics.get("expected_return", 0)
        volatility = portfolio_metrics.get("volatility", 0)
        sharpe_ratio = portfolio_metrics.get("sharpe_ratio", 0)
        
        parts = []
        
        parts.append(
            f"Portfolio optimized for {risk_profile} risk profile with "
            f"{expected_return:.1f}% expected return and {volatility:.1f}% volatility."
        )
        
        if sharpe_ratio > 1.0:
            parts.append(
                f"Strong risk-adjusted performance (Sharpe: {sharpe_ratio:.2f})."
            )
        elif sharpe_ratio > 0.5:
            parts.append(
                f"Acceptable risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})."
            )
        else:
            parts.append(
                f"Low risk-adjusted returns (Sharpe: {sharpe_ratio:.2f}) "
                "- consider alternatives."
            )
        
        if buy_count > 0 and sell_count > 0:
            parts.append(
                f"Rebalancing required: {buy_count} new positions, "
                f"{sell_count} exits."
            )
        elif buy_count > 0:
            parts.append(f"Expanding portfolio with {buy_count} new positions.")
        elif sell_count > 0:
            parts.append(f"Consolidating portfolio: exiting {sell_count} positions.")
        else:
            parts.append("Portfolio well-balanced - minimal changes needed.")
        
        return " ".join(parts)
