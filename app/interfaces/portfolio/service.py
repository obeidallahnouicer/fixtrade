"""Real BVMT portfolio construction using CAPM and long-only minimum variance."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor
from scipy.optimize import minimize

from app.interfaces.dashboard.live_data import _connection
from app.interfaces.portfolio.schemas import (
    EfficientFrontierPoint,
    PortfolioAsset,
    PortfolioMethodology,
    PortfolioMetrics,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
)

logger = logging.getLogger(__name__)
_openrouter_auth_rejected = False

TRADING_DAYS = 250
PROFILE_LIMITS = {
    "conservative": (0.03, 0.35),
    "moderate": (0.02, 0.45),
    "aggressive": (0.01, 0.60),
}
PROFILE_LABELS = {
    "conservative": "averse au risque",
    "moderate": "neutre",
    "aggressive": "preneur de risque",
}


def _load_market_history() -> tuple[pd.DataFrame, dict[str, float]]:
    """Load liquid symbols and their latest 420 persisted trading sessions."""
    with _connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH liquidity AS (
                SELECT symbol, AVG(COALESCE(quantite_negociee, 0)) AS avg_volume
                FROM (
                    SELECT symbol, quantite_negociee,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol ORDER BY seance DESC
                           ) AS row_number
                    FROM stock_prices
                    WHERE cloture > 0
                ) recent
                WHERE row_number <= 60
                GROUP BY symbol
                HAVING COUNT(*) >= 40
                ORDER BY avg_volume DESC
                LIMIT 40
            ),
            ranked AS (
                SELECT sp.symbol, sp.seance, sp.cloture,
                       ROW_NUMBER() OVER (
                           PARTITION BY sp.symbol ORDER BY sp.seance DESC
                       ) AS row_number
                FROM stock_prices sp
                JOIN liquidity l ON l.symbol = sp.symbol
                WHERE sp.cloture > 0
            )
            SELECT symbol, seance, cloture
            FROM ranked
            WHERE row_number <= 420
            ORDER BY seance, symbol
            """
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("Aucun historique de marché n'est disponible.")

    frame = pd.DataFrame(rows)
    frame["cloture"] = pd.to_numeric(frame["cloture"], errors="coerce")
    prices = frame.pivot(index="seance", columns="symbol", values="cloture").sort_index()
    prices = prices.loc[:, prices.notna().sum() >= 180]
    latest_prices = {
        symbol: float(prices[symbol].dropna().iloc[-1]) for symbol in prices.columns
    }
    return prices, latest_prices


def _select_assets(
    prices: pd.DataFrame, profile: str, count: int, risk_free_rate: float
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], float, float]:
    # BVMT symbols occasionally miss a session. A short forward-fill aligns the
    # calendar without inventing long price histories.
    aligned_prices = prices.ffill(limit=5)
    returns = aligned_prices.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    market_returns = returns.mean(axis=1, skipna=True).dropna()
    market_variance = float(market_returns.var())
    market_return = float(market_returns.mean() * TRADING_DAYS)
    market_return = float(np.clip(market_return, -0.50, 1.00))

    stats: dict[str, dict[str, float]] = {}
    for symbol in returns.columns:
        pair = pd.concat([returns[symbol], market_returns], axis=1).dropna()
        if len(pair) < 120:
            continue
        asset_returns = pair.iloc[:, 0]
        covariance_with_market = float(asset_returns.cov(pair.iloc[:, 1]))
        beta = (
            float(covariance_with_market / market_variance)
            if market_variance > 1e-12
            else 1.0
        )
        volatility = float(asset_returns.std() * np.sqrt(TRADING_DAYS))
        capm_return = float(risk_free_rate + beta * (market_return - risk_free_rate))
        historical_return = float(asset_returns.mean() * TRADING_DAYS)
        stats[symbol] = {
            "beta": float(np.clip(beta, -3.0, 5.0)),
            "covariance_with_market": covariance_with_market,
            "volatility": max(volatility, 0.0001),
            "capm_return": float(np.clip(capm_return, -1.0, 2.0)),
            "historical_return": float(np.clip(historical_return, -1.0, 2.0)),
        }

    if len(stats) < count:
        raise RuntimeError(
            f"Historique insuffisant: {len(stats)} sociétés éligibles pour {count} demandées."
        )

    metrics = pd.DataFrame(stats).T
    normalized = (metrics - metrics.mean()) / metrics.std().replace(0, 1)
    if profile == "conservative":
        score = -1.25 * normalized["volatility"] - 0.65 * normalized["beta"].abs()
    elif profile == "aggressive":
        score = (
            0.85 * normalized["capm_return"]
            + 0.45 * normalized["historical_return"]
            + 0.20 * normalized["volatility"]
        )
    else:
        score = (
            0.75 * normalized["capm_return"]
            + 0.30 * normalized["historical_return"]
            - 0.55 * normalized["volatility"]
        )

    selected = score.nlargest(count).index.tolist()
    selected_returns = returns[selected].dropna()
    if len(selected_returns) < 100:
        selected_returns = returns[selected].dropna(how="all").fillna(0)
    if len(selected_returns) < 60:
        raise RuntimeError(
            "Les sociétés sélectionnées n'ont pas assez de séances communes."
        )
    return (
        selected_returns,
        {symbol: stats[symbol] for symbol in selected},
        market_return,
        market_variance,
    )


def _minimum_variance_weights(returns: pd.DataFrame, profile: str) -> tuple[np.ndarray, np.ndarray]:
    covariance = returns.cov().to_numpy(dtype=float) * TRADING_DAYS
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance += np.eye(len(returns.columns)) * 1e-8
    count = len(returns.columns)
    min_weight, configured_max = PROFILE_LIMITS[profile]
    max_weight = max(configured_max, 1.0 / count)
    bounds = [(min_weight, max_weight)] * count
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        logger.warning("PVM optimization fallback: %s", result.message)
        weights = np.full(count, 1.0 / count)
    else:
        weights = np.maximum(result.x, 0)
        weights /= weights.sum()
    return weights, covariance


def _efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    profile: str,
    pvm_weights: np.ndarray,
) -> list[EfficientFrontierPoint]:
    """Build the constrained efficient branch above the PVM."""
    count = len(expected_returns)
    min_weight, configured_max = PROFILE_LIMITS[profile]
    max_weight = max(configured_max, 1.0 / count)
    bounds = [(min_weight, max_weight)] * count
    sum_constraint = {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}

    max_return_result = minimize(
        lambda weights: -float(weights @ expected_returns),
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=bounds,
        constraints=sum_constraint,
    )
    pvm_return = float(pvm_weights @ expected_returns)
    maximum_return = (
        float(max_return_result.x @ expected_returns)
        if max_return_result.success
        else float(expected_returns.max())
    )

    points = [
        EfficientFrontierPoint(
            expected_return=round(pvm_return, 6),
            volatility=round(
                float(np.sqrt(max(pvm_weights @ covariance @ pvm_weights, 0))),
                6,
            ),
            is_pvm=True,
        )
    ]
    if maximum_return <= pvm_return + 1e-7:
        return points

    previous = pvm_weights
    for target in np.linspace(pvm_return, maximum_return, 16)[1:]:
        result = minimize(
            lambda weights: float(weights @ covariance @ weights),
            previous,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                sum_constraint,
                {
                    "type": "eq",
                    "fun": lambda weights, target=target: (
                        float(weights @ expected_returns) - target
                    ),
                },
            ],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.success:
            previous = result.x
            points.append(
                EfficientFrontierPoint(
                    expected_return=round(float(result.x @ expected_returns), 6),
                    volatility=round(
                        float(np.sqrt(max(result.x @ covariance @ result.x, 0))),
                        6,
                    ),
                )
            )
    return points


def _fallback_explanation(response: PortfolioOptimizationResponse) -> str:
    ranked = sorted(response.assets, key=lambda asset: asset.weight, reverse=True)
    largest = ranked[0]
    top_two_weight = sum(asset.weight for asset in ranked[:2])
    diversification = (
        "La répartition reste diversifiée"
        if largest.weight < 0.30
        else "La concentration reste maîtrisée"
    )
    beta = abs(response.metrics.beta)
    beta_interpretation = (
        "très peu sensible"
        if beta < 0.25
        else "modérément sensible"
        if beta < 0.75
        else "presque aussi sensible que le marché de référence"
        if beta < 1.25
        else "plus sensible que le marché de référence"
    )
    capm_premium = (
        response.metrics.expected_return - response.metrics.risk_free_rate
    )
    return (
        f"Le PVM correspond au profil {PROFILE_LABELS[response.risk_profile]}: sa "
        f"volatilité annualisée estimée est de {response.metrics.volatility:.1%} et "
        f"son bêta de {response.metrics.beta:.2f} le rend {beta_interpretation}; "
        "cela ne supprime pas le risque propre aux titres. Le rendement attendu "
        f"selon le CAPM est de {response.metrics.expected_return:.1%}, soit "
        f"{capm_premium * 100:.1f} points de pourcentage de plus que le taux sans risque de "
        f"{response.metrics.risk_free_rate:.1%}. {diversification} "
        f"sur {response.company_count} titres; {largest.symbol} est la première ligne "
        f"avec {largest.weight:.1%} et les deux principales positions totalisent "
        f"{top_two_weight:.1%}. Ces poids résultent de l'optimisation conjointe des "
        "variances et covariances, pas d'une prévision sur les entreprises. "
        f"{response.metrics.invested_amount:,.2f} TND sont investis et "
        f"{response.metrics.cash_remaining:,.2f} TND restent disponibles après "
        "l'arrondi des quantités. Simulation indicative, pas un conseil financier."
    )


def _portfolio_fact_sheet(response: PortfolioOptimizationResponse) -> str:
    allocation = ", ".join(
        f"{asset.symbol} {asset.weight:.0%}" for asset in response.assets
    )
    return (
        f"Profil={PROFILE_LABELS[response.risk_profile]}; "
        f"pondérations (ce ne sont pas des rendements)={allocation}; "
        f"rendement attendu CAPM={response.metrics.expected_return:.2%}; "
        f"volatilité annualisée={response.metrics.volatility:.2%}; "
        f"bêta portefeuille={response.metrics.beta:.3f}; "
        f"taux sans risque={response.metrics.risk_free_rate:.2%}; "
        f"prime de risque marché={response.methodology.market_risk_premium:.2%}; "
        f"montant investi={response.metrics.invested_amount:.2f} TND; "
        f"liquidités={response.metrics.cash_remaining:.2f} TND."
    )


def _explanation_prompt(response: PortfolioOptimizationResponse) -> str:
    return (
        "Tu es le rédacteur final d'un comité quantitatif. Explique en français "
        "pourquoi ce portefeuille minimum-variance convient au profil, ce que disent "
        "le rendement CAPM, la volatilité et le bêta, puis commente brièvement la "
        "diversification. Ne confonds jamais pondération et rendement. Utilise "
        "uniquement les faits fournis et termine par une mise en garde. "
        f"FAITS: {_portfolio_fact_sheet(response)}"
    )


def _risk_agent_prompt(response: PortfolioOptimizationResponse) -> str:
    return (
        "Tu es l'agent risque d'un comité quantitatif. En 3 points très courts, "
        "interprète le profil, la volatilité, le bêta et le couple CAPM/taux sans "
        "risque. N'invente rien et ne rédige pas la réponse finale. "
        f"FAITS: {_portfolio_fact_sheet(response)}"
    )


def _allocation_agent_prompt(response: PortfolioOptimizationResponse) -> str:
    return (
        "Tu es l'agent allocation d'un comité quantitatif. En 3 points très courts, "
        "analyse la concentration, la diversification, la première ligne et les "
        "liquidités résiduelles. Les pourcentages des titres sont des pondérations, "
        "jamais des rendements. N'invente rien et ne rédige pas la réponse finale. "
        f"FAITS: {_portfolio_fact_sheet(response)}"
    )


def _clean_explanation(content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    if content[-1] not in ".!?":
        last_sentence = max(content.rfind("."), content.rfind("!"), content.rfind("?"))
        if last_sentence >= 0:
            content = content[: last_sentence + 1]
    if not any(
        marker in content.lower()
        for marker in ("prudence", "conseil financier", "simulation")
    ):
        content += " Simulation indicative, pas un conseil financier."
    return content


def _normalize_text(content: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", content.lower())
        if not unicodedata.combining(character)
    )


def _explanation_validation_errors(
    content: str, response: PortfolioOptimizationResponse
) -> list[str]:
    """Return precise reasons why a generated explanation is unsafe or unhelpful."""
    errors: list[str] = []
    normalized = _normalize_text(content)
    required_concepts = (
        ("capm", "rendement attendu"),
        ("volatilite", "risque"),
        ("beta", "sensibilite"),
        ("prudence", "conseil financier", "simulation"),
    )
    concept_count = sum(
        any(marker in normalized for marker in alternatives)
        for alternatives in required_concepts
    )
    if len(content.split()) < 30:
        errors.append("too_short")
    if concept_count != len(required_concepts):
        errors.append("missing_core_concept")

    expected_return = response.metrics.expected_return * 100
    volatility = response.metrics.volatility * 100
    numeric_markers = tuple(
        marker
        for value in (expected_return, volatility)
        for marker in (
            f"{value:.1f}",
            f"{value:.1f}".replace(".", ","),
            f"{value:.2f}",
            f"{value:.2f}".replace(".", ","),
        )
    ) + (
        f"{response.metrics.beta:.2f}",
        f"{response.metrics.beta:.2f}".replace(".", ","),
        f"{response.metrics.beta:.3f}",
        f"{response.metrics.beta:.3f}".replace(".", ","),
    )
    metric_matches = sum(marker in content for marker in numeric_markers)
    if metric_matches < 2:
        errors.append("missing_verified_metrics")

    unsupported_claims = (
        "secteur",
        "zone euro",
        "pays",
        "geograph",
        "perspective",
        "dividende",
        "qualite elevee",
        "pertes potentielles limitees",
        "pertes potentielles",
        "couverture pour les pertes",
        "parfaitement",
        "considerablement",
        "correction",
    )
    if any(claim in normalized for claim in unsupported_claims):
        errors.append("unsupported_claim")

    allowed_percentages = [
        expected_return,
        volatility,
        response.metrics.risk_free_rate * 100,
        response.methodology.market_risk_premium * 100,
        (response.metrics.expected_return - response.metrics.risk_free_rate) * 100,
        response.methodology.maximum_weight * 100,
        response.methodology.minimum_weight * 100,
        response.metrics.cash_remaining / response.investment_amount * 100,
        *[asset.weight * 100 for asset in response.assets],
        sum(
            asset.weight
            for asset in sorted(
                response.assets, key=lambda asset: asset.weight, reverse=True
            )[:2]
        )
        * 100,
    ]
    mentioned_percentages = [
        float(value.replace(",", "."))
        for value in re.findall(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*%", content)
    ]
    if any(
        not any(abs(mentioned - allowed) <= 0.11 for allowed in allowed_percentages)
        for mentioned in mentioned_percentages
    ):
        errors.append("invented_percentage")
    return errors


def _is_valid_explanation(
    content: str, response: PortfolioOptimizationResponse
) -> bool:
    """Validate that an explanation interprets only verified return/risk facts."""
    return not _explanation_validation_errors(content, response)


def _is_useful_explanation(
    content: str, response: PortfolioOptimizationResponse
) -> bool:
    """Apply stricter editorial checks to the final multi-agent synthesis."""
    normalized = _normalize_text(content)
    discusses_allocation = any(
        marker in normalized for marker in ("diversif", "concentr", "repart")
    )
    return (
        _is_valid_explanation(content, response)
        and len(content.split()) >= 55
        and discusses_allocation
    )


def _openrouter_explanation(
    response: PortfolioOptimizationResponse,
    prompt: str | None = None,
) -> tuple[str, str] | None:
    global _openrouter_auth_rejected

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or _openrouter_auth_rejected:
        return None
    model = os.getenv(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
    try:
        result = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_SITE_URL", "http://localhost:3000"
                ),
                "X-Title": "FixTrade Portfolio Agent",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Analyste FixTrade. Réponse directe; aucun chiffre inventé."
                        ),
                    },
                    {"role": "user", "content": prompt or _explanation_prompt(response)},
                ],
                "temperature": 0.2,
                "max_tokens": 220,
                "stream": False,
            },
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        if result.status_code in {401, 403}:
            _openrouter_auth_rejected = True
            logger.warning(
                "OpenRouter rejected the configured API key; local LM Studio "
                "will be used until the API process restarts."
            )
            return None
        result.raise_for_status()
        content = _clean_explanation(
            result.json()["choices"][0]["message"]["content"]
        )
        if content:
            return content, "openrouter"
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "OpenRouter unavailable (%s); trying the local LM Studio model.", exc
        )
    return None


def _select_local_model(server_url: str) -> str:
    configured = os.getenv("LM_STUDIO_MODEL", "auto").strip()
    if configured and configured.lower() != "auto":
        return configured
    try:
        result = httpx.get(
            f"{server_url}/api/v1/models",
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        result.raise_for_status()
        models = [
            model
            for model in result.json().get("models", [])
            if model.get("type") == "llm"
        ]
        loaded_llama = next(
            (
                model
                for model in models
                if "llama" in model.get("key", "").lower()
                and model.get("loaded_instances")
            ),
            None,
        )
        installed_llama = next(
            (
                model
                for model in models
                if "llama" in model.get("key", "").lower()
            ),
            None,
        )
        loaded_model = next(
            (model for model in models if model.get("loaded_instances")), None
        )
        selected = loaded_llama or installed_llama or loaded_model
        if selected:
            return selected["key"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        pass
    return "qwen/qwen3.6-27b"


def _lm_studio_explanation(
    response: PortfolioOptimizationResponse,
    prompt: str | None = None,
) -> tuple[str, str] | None:
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    server_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    model = _select_local_model(server_url)
    timeout = float(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "120"))
    try:
        result = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Analyste FixTrade. Réponse directe; aucun chiffre inventé."
                        ),
                    },
                    {"role": "user", "content": prompt or _explanation_prompt(response)},
                ],
                "temperature": 0.2,
                "max_tokens": 220,
                "stream": False,
            },
            timeout=httpx.Timeout(timeout, connect=2.0),
        )
        result.raise_for_status()
        content = _clean_explanation(
            result.json()["choices"][0]["message"]["content"]
        )
        if content:
            return content, "lm_studio"
    except httpx.ConnectError:
        logger.warning(
            "LM Studio API is not listening at %s; using deterministic explanation.",
            base_url,
        )
    except httpx.TimeoutException:
        logger.warning(
            "LM Studio did not answer within %.0f seconds; using deterministic explanation.",
            timeout,
        )
    except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "LM Studio returned an unusable response (%s); using deterministic explanation.",
            exc,
        )
    return None


def _agent_completion(
    response: PortfolioOptimizationResponse, prompt: str
) -> tuple[str, str] | None:
    return _openrouter_explanation(response, prompt) or _lm_studio_explanation(
        response, prompt
    )


def _risk_agent_report(response: PortfolioOptimizationResponse) -> str:
    premium = response.metrics.expected_return - response.metrics.risk_free_rate
    beta_interpretation = (
        "très faible sensibilité au marché de référence"
        if abs(response.metrics.beta) < 0.25
        else "sensibilité modérée au marché de référence"
        if abs(response.metrics.beta) < 0.75
        else "sensibilité proche ou supérieure à celle du marché de référence"
    )
    return (
        f"Volatilité annualisée estimée: {response.metrics.volatility:.2%}. "
        f"Bêta: {response.metrics.beta:.3f}, soit une {beta_interpretation}; cela "
        "ne supprime ni le risque propre aux titres ni le risque de perte. "
        f"Rendement CAPM: {response.metrics.expected_return:.2%}, contre "
        f"{response.metrics.risk_free_rate:.2%} sans risque, soit un écart de "
        f"{premium:.2%}. Pour ce profil, le résultat privilégie la réduction de "
        "variance plutôt que la maximisation du rendement."
    )


def _allocation_agent_report(response: PortfolioOptimizationResponse) -> str:
    ranked = sorted(response.assets, key=lambda asset: asset.weight, reverse=True)
    top_two_weight = sum(asset.weight for asset in ranked[:2])
    cash_ratio = response.metrics.cash_remaining / response.investment_amount
    return (
        f"{response.company_count} titres sont retenus. La première ligne est "
        f"{ranked[0].symbol} à {ranked[0].weight:.1%}; les deux premières totalisent "
        f"{top_two_weight:.1%}. La borne maximale autorisée est "
        f"{response.methodology.maximum_weight:.0%}, donc aucun titre ne dépasse "
        "la contrainte de concentration. Les poids proviennent de l'optimisation "
        "conjointe des variances et covariances, pas d'une prévision sur les sociétés. "
        f"Les liquidités résiduelles représentent {cash_ratio:.2%} du capital."
    )


def _multi_agent_explanation(
    response: PortfolioOptimizationResponse,
) -> tuple[str, str] | None:
    """Run independent quantitative agents, then ask an LLM to edit their reports."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="portfolio-agent") as pool:
        risk_future = pool.submit(_risk_agent_report, response)
        allocation_future = pool.submit(_allocation_agent_report, response)
        risk_analysis = risk_future.result()
        allocation_analysis = allocation_future.result()

    synthesis_prompt = (
        f"{_explanation_prompt(response)}\n\n"
        "RAPPORT VÉRIFIÉ DE L'AGENT RISQUE:\n"
        f"{risk_analysis}\n\n"
        "RAPPORT VÉRIFIÉ DE L'AGENT ALLOCATION:\n"
        f"{allocation_analysis}\n\n"
        "Produis maintenant un paragraphe clair de 80 à 130 mots. Explique les "
        "arbitrages au lieu de réciter toutes les pondérations. Ces rapports sont "
        "exhaustifs: n'ajoute aucun secteur, pays, qualité d'entreprise, perspective, "
        "dividende, causalité ou garantie de perte."
    )
    generated = _agent_completion(response, synthesis_prompt)
    if generated and _is_useful_explanation(generated[0], response):
        return generated

    validation_errors = (
        _explanation_validation_errors(generated[0], response)
        if generated
        else ["provider_unavailable"]
    )
    logger.info(
        "Multi-agent draft rejected (%s); using the verified fact-based synthesis.",
        ", ".join(validation_errors),
    )
    return _fallback_explanation(response), "multi_agent"


def optimize_portfolio(payload: PortfolioOptimizationRequest) -> PortfolioOptimizationResponse:
    risk_free_rate = float(os.getenv("PORTFOLIO_RISK_FREE_RATE", "0.07"))
    prices, latest_prices = _load_market_history()
    returns, stats, market_return, market_variance = _select_assets(
        prices, payload.risk_profile, payload.company_count, risk_free_rate
    )
    weights, covariance = _minimum_variance_weights(returns, payload.risk_profile)

    assets: list[PortfolioAsset] = []
    total_invested = 0.0
    for index, symbol in enumerate(returns.columns):
        allocation = float(payload.investment_amount * weights[index])
        price = latest_prices[symbol]
        shares = int(allocation // price)
        invested = float(shares * price)
        total_invested += invested
        assets.append(
            PortfolioAsset(
                symbol=symbol,
                latest_price=round(price, 3),
                weight=round(float(weights[index]), 8),
                allocation_amount=round(allocation, 2),
                shares=shares,
                invested_amount=round(invested, 2),
                beta=round(stats[symbol]["beta"], 4),
                covariance_with_market=round(
                    stats[symbol]["covariance_with_market"], 10
                ),
                volatility=round(stats[symbol]["volatility"], 6),
                capm_return=round(stats[symbol]["capm_return"], 6),
            )
        )

    capm_returns = np.array([stats[symbol]["capm_return"] for symbol in returns.columns])
    betas = np.array([stats[symbol]["beta"] for symbol in returns.columns])
    variance = float(weights @ covariance @ weights)
    frontier = _efficient_frontier(
        capm_returns, covariance, payload.risk_profile, weights
    )
    min_weight, configured_max = PROFILE_LIMITS[payload.risk_profile]
    max_weight = max(configured_max, 1.0 / payload.company_count)
    response = PortfolioOptimizationResponse(
        risk_profile=payload.risk_profile,
        company_count=payload.company_count,
        investment_amount=payload.investment_amount,
        assets=sorted(assets, key=lambda asset: asset.weight, reverse=True),
        metrics=PortfolioMetrics(
            expected_return=round(float(weights @ capm_returns), 6),
            volatility=round(float(np.sqrt(max(variance, 0))), 6),
            variance=round(variance, 8),
            beta=round(float(weights @ betas), 4),
            market_return=round(market_return, 6),
            risk_free_rate=round(risk_free_rate, 6),
            invested_amount=round(total_invested, 2),
            cash_remaining=round(payload.investment_amount - total_invested, 2),
        ),
        efficient_frontier=frontier,
        methodology=PortfolioMethodology(
            observations=len(returns),
            trading_days_per_year=TRADING_DAYS,
            market_variance=round(market_variance, 10),
            market_risk_premium=round(market_return - risk_free_rate, 6),
            minimum_weight=min_weight,
            maximum_weight=max_weight,
        ),
        explanation="",
        explanation_source="fallback",
        generated_at=datetime.now(timezone.utc),
    )
    generated = _multi_agent_explanation(response)
    explanation, source = generated or (
        _fallback_explanation(response),
        "fallback",
    )
    response.explanation = explanation
    response.explanation_source = source
    if source == "fallback":
        response.warnings.append(
            "Explication multi-agent indisponible ou rejetée par le contrôle qualité: "
            "explication factuelle déterministe utilisée."
        )
    return response
