"""Unit tests for portfolio explanation quality validation."""

from datetime import datetime, timezone

import pytest

from app.interfaces.portfolio.schemas import (
    EfficientFrontierPoint,
    PortfolioAsset,
    PortfolioMethodology,
    PortfolioMetrics,
    PortfolioOptimizationResponse,
)
from app.interfaces.portfolio.service import (
    _fallback_explanation,
    _is_valid_explanation,
)


@pytest.fixture
def portfolio_response() -> PortfolioOptimizationResponse:
    assets = [
        PortfolioAsset(
            symbol="BIAT",
            latest_price=98.0,
            weight=0.45,
            allocation_amount=4500.0,
            shares=45,
            invested_amount=4410.0,
            beta=0.75,
            covariance_with_market=0.00012,
            volatility=0.12,
            capm_return=0.10,
        ),
        PortfolioAsset(
            symbol="SFBT",
            latest_price=13.0,
            weight=0.35,
            allocation_amount=3500.0,
            shares=269,
            invested_amount=3497.0,
            beta=0.90,
            covariance_with_market=0.00018,
            volatility=0.16,
            capm_return=0.12,
        ),
        PortfolioAsset(
            symbol="SAH",
            latest_price=9.0,
            weight=0.20,
            allocation_amount=2000.0,
            shares=222,
            invested_amount=1998.0,
            beta=1.10,
            covariance_with_market=0.00024,
            volatility=0.21,
            capm_return=0.15,
        ),
    ]
    return PortfolioOptimizationResponse(
        risk_profile="moderate",
        company_count=3,
        investment_amount=10_000.0,
        assets=assets,
        metrics=PortfolioMetrics(
            expected_return=0.114,
            volatility=0.137,
            variance=0.018769,
            beta=0.82,
            market_return=0.13,
            risk_free_rate=0.07,
            invested_amount=9905.0,
            cash_remaining=95.0,
        ),
        efficient_frontier=[
            EfficientFrontierPoint(
                expected_return=0.114,
                volatility=0.137,
                is_pvm=True,
            )
        ],
        methodology=PortfolioMethodology(
            observations=250,
            trading_days_per_year=250,
            market_variance=0.021,
            market_risk_premium=0.06,
            minimum_weight=0.02,
            maximum_weight=0.45,
        ),
        explanation="",
        explanation_source="fallback",
        generated_at=datetime.now(timezone.utc),
    )


def test_accepts_explanation_that_interprets_return_and_risk(
    portfolio_response: PortfolioOptimizationResponse,
) -> None:
    explanation = (
        "Le rendement CAPM attendu de 11,4 % rémunère une volatilité annualisée "
        "de 13,7 %. Avec un bêta de 0,82, le portefeuille devrait être moins "
        "sensible que le marché, sans supprimer le risque de perte : cette "
        "simulation ne constitue pas un conseil financier."
    )

    assert _is_valid_explanation(explanation, portfolio_response)


@pytest.mark.parametrize(
    "explanation",
    [
        "Ce portefeuille est bien diversifié et adapté à votre profil. Prudence.",
        (
            "Cette allocation optimisée offre un bon équilibre entre rendement "
            "et risque. Simulation indicative, pas un conseil financier."
        ),
    ],
)
def test_rejects_hollow_generic_explanation(
    explanation: str,
    portfolio_response: PortfolioOptimizationResponse,
) -> None:
    assert not _is_valid_explanation(explanation, portfolio_response)


def test_rejects_weight_recitation_without_metric_interpretation(
    portfolio_response: PortfolioOptimizationResponse,
) -> None:
    explanation = (
        "BIAT représente 45 %, SFBT 35 % et SAH 20 % du portefeuille. "
        "Les pondérations sont donc BIAT 45 %, SFBT 35 % et SAH 20 % ; "
        "prudence, ceci reste une simulation."
    )

    assert not _is_valid_explanation(explanation, portfolio_response)


def test_rejects_invented_portfolio_percentages(
    portfolio_response: PortfolioOptimizationResponse,
) -> None:
    explanation = (
        "Le rendement CAPM attendu est de 18,2 % pour une volatilité de 4,9 %. "
        "Le bêta de 0,82 traduit une sensibilité modérée au marché et la "
        "diversification réduit certains risques spécifiques. Cette simulation "
        "indicative ne constitue pas un conseil financier."
    )

    assert not _is_valid_explanation(explanation, portfolio_response)


def test_fallback_does_not_call_near_market_beta_low(
    portfolio_response: PortfolioOptimizationResponse,
) -> None:
    portfolio_response.metrics.beta = 0.91

    explanation = _fallback_explanation(portfolio_response)

    assert "presque aussi sensible que le marché de référence" in explanation
    assert "faible sensibilité" not in explanation
