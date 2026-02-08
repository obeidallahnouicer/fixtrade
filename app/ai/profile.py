"""
User Risk Profile Management.

Defines risk profiles and manages user investment preferences:
- Conservative: Low risk, stable returns, focus on capital preservation
- Moderate: Balanced risk/reward, diversified portfolio
- Aggressive: High risk tolerance, growth-focused

Each profile determines:
- Maximum position sizes
- Asset allocation limits
- Stop-loss thresholds
- Recommended holding periods
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime
import logging

from app.ai.config import ai_settings

logger = logging.getLogger(__name__)


class RiskProfile(str, Enum):
    """Investment risk profiles."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class ProfileCharacteristics:
    """Characteristics of a risk profile."""
    max_position_size: float  # % of portfolio
    max_equity_allocation: float  # % in stocks vs cash
    stop_loss_threshold: float  # % loss trigger
    min_holding_days: int
    preferred_liquidity: str  # high/medium/low
    description: str


class UserProfileManager:
    """Manages user risk profiles and investment preferences."""
    
    PROFILE_CONFIGS: Dict[RiskProfile, ProfileCharacteristics] = {
        RiskProfile.CONSERVATIVE: ProfileCharacteristics(
            max_position_size=ai_settings.conservative_max_position_size,
            max_equity_allocation=ai_settings.conservative_max_equity_allocation,
            stop_loss_threshold=ai_settings.stop_loss_conservative,
            min_holding_days=5,
            preferred_liquidity="high",
            description=(
                "Profil conservateur : Priorité à la préservation du capital. "
                "Investissements dans des valeurs stables et liquides avec une "
                "diversification maximale."
            )
        ),
        RiskProfile.MODERATE: ProfileCharacteristics(
            max_position_size=ai_settings.moderate_max_position_size,
            max_equity_allocation=ai_settings.moderate_max_equity_allocation,
            stop_loss_threshold=ai_settings.stop_loss_moderate,
            min_holding_days=3,
            preferred_liquidity="medium",
            description=(
                "Profil modéré : Équilibre entre risque et rendement. "
                "Diversification avec une allocation équilibrée entre "
                "valeurs de croissance et valeurs défensives."
            )
        ),
        RiskProfile.AGGRESSIVE: ProfileCharacteristics(
            max_position_size=ai_settings.aggressive_max_position_size,
            max_equity_allocation=ai_settings.aggressive_max_equity_allocation,
            stop_loss_threshold=ai_settings.stop_loss_aggressive,
            min_holding_days=1,
            preferred_liquidity="low",
            description=(
                "Profil agressif : Recherche de rendement maximal. "
                "Tolérance élevée à la volatilité avec concentration "
                "sur les opportunités de croissance."
            )
        ),
    }
    
    def __init__(self, default_profile: RiskProfile = RiskProfile.MODERATE):
        """Initialize profile manager."""
        self.default_profile = default_profile
        self.user_profiles: Dict[str, RiskProfile] = {}
        logger.info(f"UserProfileManager initialized with default: {default_profile}")
    
    def get_characteristics(self, profile: RiskProfile) -> ProfileCharacteristics:
        """Get characteristics for a risk profile."""
        return self.PROFILE_CONFIGS[profile]
    
    def set_user_profile(self, user_id: str, profile: RiskProfile) -> None:
        """Set risk profile for a user."""
        self.user_profiles[user_id] = profile
        logger.info(f"Set profile {profile} for user {user_id}")
    
    def get_user_profile(self, user_id: str) -> RiskProfile:
        """Get user's risk profile, or default if not set."""
        return self.user_profiles.get(user_id, self.default_profile)
    
    def recommend_profile(self, questionnaire_data: Dict[str, Any]) -> RiskProfile:
        """
        Recommend a risk profile based on questionnaire responses.
        
        Questionnaire should include:
        - age: int
        - investment_horizon: int (years)
        - income_stability: str (high/medium/low)
        - investment_experience: str (beginner/intermediate/expert)
        - loss_tolerance: int (1-5 scale)
        - financial_goals: str (preservation/growth/aggressive_growth)
        """
        score = 0
        
        # Age factor
        age = questionnaire_data.get("age", 30)
        if age < 30:
            score += 2
        elif age < 50:
            score += 1
        
        # Investment horizon
        horizon = questionnaire_data.get("investment_horizon", 5)
        if horizon > 10:
            score += 2
        elif horizon > 5:
            score += 1
        
        # Loss tolerance (1-5)
        loss_tolerance = questionnaire_data.get("loss_tolerance", 3)
        score += (loss_tolerance - 3)
        
        # Experience
        experience = questionnaire_data.get("investment_experience", "beginner")
        if experience == "expert":
            score += 2
        elif experience == "intermediate":
            score += 1
        
        # Financial goals
        goals = questionnaire_data.get("financial_goals", "growth")
        if goals == "aggressive_growth":
            score += 2
        elif goals == "growth":
            score += 1
        elif goals == "preservation":
            score -= 1
        
        # Determine profile based on score
        if score <= 2:
            recommended = RiskProfile.CONSERVATIVE
        elif score <= 5:
            recommended = RiskProfile.MODERATE
        else:
            recommended = RiskProfile.AGGRESSIVE
        
        logger.info(
            f"Profile recommendation: {recommended} (score: {score}) "
            f"for questionnaire: {questionnaire_data}"
        )
        
        return recommended
    
    def validate_trade(
        self,
        profile: RiskProfile,
        position_size: float,
        current_equity_allocation: float
    ) -> tuple[bool, str]:
        """
        Validate if a trade is appropriate for the risk profile.
        
        Returns:
            (is_valid, reason)
        """
        chars = self.get_characteristics(profile)
        
        # Check position size
        if position_size > chars.max_position_size:
            return False, (
                f"Position trop importante : {position_size:.1%} "
                f"(max: {chars.max_position_size:.1%})"
            )
        
        # Check equity allocation
        if current_equity_allocation > chars.max_equity_allocation:
            return False, (
                f"Allocation en actions trop élevée : {current_equity_allocation:.1%} "
                f"(max: {chars.max_equity_allocation:.1%})"
            )
        
        return True, "Trade conforme au profil de risque"
