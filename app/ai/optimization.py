"""
Portfolio Optimization Engine.

Implements:
- Minimum Variance Portfolio (PMV)
- Efficient Frontier construction
- CAPM/MEDAF expected returns
- Diversification constraints
- Risk-adjusted portfolio construction

All calculations are deterministic and config-driven.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from app.ai.profile import RiskProfile

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: float = 0.3
    target_volatility: Optional[float] = None
    allow_short: bool = False


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""
    symbols: List[str]
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float


@dataclass
class EfficientFrontierPoint:
    """Single point on efficient frontier."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


class PortfolioOptimizer:
    """
    Portfolio optimization engine using Modern Portfolio Theory.
    
    Implements minimum variance, maximum Sharpe, and efficient frontier.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 250
    ):
        """
        Initialize optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (TND bonds)
            trading_days_per_year: Trading days for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def calculate_covariance_matrix(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate annualized covariance matrix.
        
        Args:
            returns: DataFrame of daily returns (symbols as columns)
        
        Returns:
            Annualized covariance matrix
        """
        cov_matrix = returns.cov()
        annualized_cov = cov_matrix * self.trading_days_per_year
        return annualized_cov.values
    
    def calculate_expected_returns(
        self,
        returns: pd.DataFrame,
        predictions: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Calculate expected returns.
        
        Uses predictions if available, otherwise historical mean.
        
        Args:
            returns: DataFrame of daily returns
            predictions: Dict of symbol -> predicted return (annual)
        
        Returns:
            Array of expected annual returns
        """
        if predictions:
            symbols = returns.columns.tolist()
            expected = np.array([
                predictions.get(sym, returns[sym].mean() * self.trading_days_per_year)
                for sym in symbols
            ])
        else:
            expected = returns.mean().values * self.trading_days_per_year
        
        return expected
    
    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
        
        Returns:
            (expected_return, volatility, sharpe_ratio)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0 else 0.0
        )
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        portfolio_volatility: float
    ) -> float:
        """
        Calculate diversification ratio.
        
        DR = (weighted avg of individual volatilities) / portfolio volatility
        Higher is better (more diversification benefit).
        
        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            portfolio_volatility: Portfolio volatility
        
        Returns:
            Diversification ratio
        """
        weighted_vol = np.dot(weights, volatilities)
        
        if portfolio_volatility > 0:
            return weighted_vol / portfolio_volatility
        
        return 1.0
    
    def minimum_variance_portfolio(
        self,
        returns: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> PortfolioWeights:
        """
        Calculate minimum variance portfolio (PMV).
        
        Args:
            returns: DataFrame of daily returns
            constraints: Optimization constraints
        
        Returns:
            PortfolioWeights with optimal allocation
        """
        n_assets = len(returns.columns)
        symbols = returns.columns.tolist()
        
        cov_matrix = self.calculate_covariance_matrix(returns)
        expected_returns = self.calculate_expected_returns(returns)
        
        def objective(weights):
            """Minimize portfolio variance."""
            variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return variance
        
        # Constraints: weights sum to 1
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds for each weight
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Minimum variance optimization failed: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate metrics
        exp_return, volatility, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, expected_returns, cov_matrix
        )
        
        individual_vols = np.sqrt(np.diag(cov_matrix))
        div_ratio = self.calculate_diversification_ratio(
            optimal_weights, individual_vols, volatility
        )
        
        return PortfolioWeights(
            symbols=symbols,
            weights=optimal_weights,
            expected_return=exp_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio
        )
    
    def maximum_sharpe_portfolio(
        self,
        returns: pd.DataFrame,
        constraints: OptimizationConstraints,
        predictions: Optional[Dict[str, float]] = None
    ) -> PortfolioWeights:
        """
        Calculate maximum Sharpe ratio portfolio.
        
        Args:
            returns: DataFrame of daily returns
            constraints: Optimization constraints
            predictions: Optional predicted returns
        
        Returns:
            PortfolioWeights with optimal allocation
        """
        n_assets = len(returns.columns)
        symbols = returns.columns.tolist()
        
        cov_matrix = self.calculate_covariance_matrix(returns)
        expected_returns = self.calculate_expected_returns(returns, predictions)
        
        def objective(weights):
            """Minimize negative Sharpe ratio."""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility == 0:
                return 1e10
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Maximum Sharpe optimization failed: {result.message}")
        
        optimal_weights = result.x
        
        exp_return, volatility, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, expected_returns, cov_matrix
        )
        
        individual_vols = np.sqrt(np.diag(cov_matrix))
        div_ratio = self.calculate_diversification_ratio(
            optimal_weights, individual_vols, volatility
        )
        
        return PortfolioWeights(
            symbols=symbols,
            weights=optimal_weights,
            expected_return=exp_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio
        )
    
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        constraints: OptimizationConstraints,
        n_points: int = 50,
        predictions: Optional[Dict[str, float]] = None
    ) -> List[EfficientFrontierPoint]:
        """
        Calculate efficient frontier.
        
        Args:
            returns: DataFrame of daily returns
            constraints: Optimization constraints
            n_points: Number of frontier points
            predictions: Optional predicted returns
        
        Returns:
            List of efficient frontier points
        """
        n_assets = len(returns.columns)
        symbols = returns.columns.tolist()
        
        cov_matrix = self.calculate_covariance_matrix(returns)
        expected_returns = self.calculate_expected_returns(returns, predictions)
        
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_points = []
        
        for target_return in target_returns:
            def objective(weights):
                """Minimize portfolio variance."""
                variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return variance
            
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, tr=target_return: np.dot(w, expected_returns) - tr}
            ]
            
            bounds = tuple(
                (constraints.min_weight, constraints.max_weight)
                for _ in range(n_assets)
            )
            
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if result.success:
                weights = result.x
                exp_return, volatility, sharpe = self.calculate_portfolio_metrics(
                    weights, expected_returns, cov_matrix
                )
                
                point = EfficientFrontierPoint(
                    expected_return=exp_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe,
                    weights={sym: float(w) for sym, w in zip(symbols, weights)}
                )
                frontier_points.append(point)
        
        return frontier_points
    
    def optimize_for_profile(
        self,
        returns: pd.DataFrame,
        risk_profile: RiskProfile,
        predictions: Optional[Dict[str, float]] = None
    ) -> PortfolioWeights:
        """
        Optimize portfolio for specific risk profile.
        
        Args:
            returns: DataFrame of daily returns
            risk_profile: User risk profile
            predictions: Optional predicted returns
        
        Returns:
            PortfolioWeights optimized for profile
        """
        if risk_profile == RiskProfile.CONSERVATIVE:
            constraints = OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.15,
                max_sector_weight=0.25
            )
            portfolio = self.minimum_variance_portfolio(returns, constraints)
            
        elif risk_profile == RiskProfile.MODERATE:
            constraints = OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.25,
                max_sector_weight=0.35
            )
            portfolio = self.maximum_sharpe_portfolio(returns, constraints, predictions)
            
        else:  # AGGRESSIVE
            constraints = OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.40,
                max_sector_weight=0.50
            )
            portfolio = self.maximum_sharpe_portfolio(returns, constraints, predictions)
        
        logger.info(
            f"Optimized portfolio for {risk_profile}: "
            f"Return={portfolio.expected_return:.2%}, "
            f"Vol={portfolio.expected_volatility:.2%}, "
            f"Sharpe={portfolio.sharpe_ratio:.2f}"
        )
        
        return portfolio


class CAPMCalculator:
    """
    CAPM (Capital Asset Pricing Model) / MEDAF calculator.
    
    Calculates expected returns based on systematic risk (beta).
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        market_premium: float = 0.08
    ):
        """
        Initialize CAPM calculator.
        
        Args:
            risk_free_rate: Risk-free rate (TND bonds)
            market_premium: Market risk premium
        """
        self.risk_free_rate = risk_free_rate
        self.market_premium = market_premium
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate beta (systematic risk).
        
        Beta = Cov(asset, market) / Var(market)
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
        
        Returns:
            Beta coefficient
        """
        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_expected_return(
        self,
        beta: float
    ) -> float:
        """
        Calculate CAPM expected return.
        
        E(R) = Rf + Beta * (Rm - Rf)
        
        Args:
            beta: Asset beta
        
        Returns:
            Expected annual return
        """
        expected_return = self.risk_free_rate + beta * self.market_premium
        return expected_return
    
    def calculate_all_betas(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate beta for all assets.
        
        Args:
            returns: DataFrame of asset returns
            market_returns: Market return series
        
        Returns:
            Dict of symbol -> beta
        """
        betas = {}
        
        for symbol in returns.columns:
            beta = self.calculate_beta(returns[symbol], market_returns)
            betas[symbol] = beta
        
        return betas
    
    def calculate_all_expected_returns(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate CAPM expected returns for all assets.
        
        Args:
            returns: DataFrame of asset returns
            market_returns: Market return series
        
        Returns:
            Dict of symbol -> expected annual return
        """
        betas = self.calculate_all_betas(returns, market_returns)
        
        expected_returns = {
            symbol: self.calculate_expected_return(beta)
            for symbol, beta in betas.items()
        }
        
        return expected_returns
