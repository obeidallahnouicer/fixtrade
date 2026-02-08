"""
Use case: Get a trade recommendation for a BVMT-listed symbol.

Input: GetRecommendationQuery (symbol, portfolio_id)
Output: RecommendationResult
Side effects: None.
Failure cases: SymbolNotFoundError, PortfolioNotFoundError.
"""

import logging

from app.application.trading.dtos import GetRecommendationQuery, RecommendationResult
from app.domain.trading.errors import PortfolioNotFoundError
from app.domain.trading.ports import DecisionEnginePort, PortfolioRepository

logger = logging.getLogger(__name__)


class GetRecommendationUseCase:
    """Orchestrates generating a trade recommendation.

    Verifies the portfolio exists, then delegates to the
    DecisionEnginePort for the actual recommendation logic.
    """

    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        decision_port: DecisionEnginePort,
    ) -> None:
        self._portfolio_repo = portfolio_repo
        self._decision_port = decision_port

    def execute(self, query: GetRecommendationQuery) -> RecommendationResult:
        """Run the recommendation use case.

        Args:
            query: The recommendation request with symbol and portfolio ID.

        Returns:
            A trade recommendation with action and reasoning.

        Raises:
            PortfolioNotFoundError: If the portfolio does not exist.
        """
        logger.info(
            "Generating recommendation for symbol=%s, portfolio=%s",
            query.symbol,
            query.portfolio_id,
        )

        portfolio = self._portfolio_repo.get_by_id(query.portfolio_id)
        if portfolio is None:
            raise PortfolioNotFoundError(str(query.portfolio_id))

        # TODO: call the decision engine port and map result to DTO
        recommendation = self._decision_port.recommend(
            symbol=query.symbol,
            portfolio_id=query.portfolio_id,
        )

        return RecommendationResult(
            symbol=recommendation.symbol,
            action=recommendation.action.value,
            confidence=recommendation.confidence,
            reasoning=recommendation.reasoning,
        )
