"""Cost tracking service for LLM API calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlmodel import Session, func, select

from llm_tournament.models import LLMCall
from llm_tournament.services.storage.repository import AsyncRepository

from .pricing import PricingService

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = structlog.get_logger()


class CostTracker(AsyncRepository):
    """Tracks and computes costs for LLM API calls.

    Usage:
        tracker = CostTracker(pricing_service, engine)
        cost = await tracker.record_call(model, response, role="writer")
        breakdown = await tracker.get_cost_breakdown()
    """

    def __init__(self, pricing_service: PricingService, engine: Engine) -> None:
        """Initialize cost tracker.

        Args:
            pricing_service: Service providing model pricing data.
            engine: SQLAlchemy engine for database operations.
        """
        super().__init__(engine)
        self._pricing = pricing_service

    async def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        role: str,
        topic_slug: str | None = None,
    ) -> float:
        """Compute cost and store call in database.

        Args:
            model: OpenRouter model ID.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            total_tokens: Total tokens used.
            role: Call role ("writer", "judge", "critic", "analysis").
            topic_slug: Optional topic context.

        Returns:
            Computed cost in USD.
        """
        # Compute cost
        cost_usd = self._pricing.compute_cost(model, prompt_tokens, completion_tokens)
        if cost_usd is None:
            logger.warning("cost_unknown_model", model=model)
            cost_usd = 0.0

        call = LLMCall(
            model=model,
            role=role,
            topic_slug=topic_slug,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

        def _save(session: Session) -> None:
            session.add(call)
            session.commit()

        await self._run_session(_save)

        logger.debug(
            "cost_recorded",
            model=model,
            role=role,
            cost_usd=cost_usd,
            tokens=total_tokens,
        )
        return cost_usd

    async def get_cost_breakdown(self) -> dict[str, float]:
        """Get total spend grouped by role.

        Returns:
            Dictionary mapping role to total USD spend.
        """

        def _query(session: Session) -> dict[str, float]:
            statement = select(LLMCall.role, func.sum(LLMCall.cost_usd)).group_by(LLMCall.role)
            results = session.exec(statement).all()
            return {role: float(cost) for role, cost in results}

        return await self._run_session(_query)

    async def get_model_costs(self) -> dict[str, float]:
        """Get total spend grouped by model.

        Returns:
            Dictionary mapping model ID to total USD spend.
        """

        def _query(session: Session) -> dict[str, float]:
            statement = select(LLMCall.model, func.sum(LLMCall.cost_usd)).group_by(LLMCall.model)
            results = session.exec(statement).all()
            return {model: float(cost) for model, cost in results}

        return await self._run_session(_query)

    async def get_total_cost(self) -> float:
        """Get total spend across all calls.

        Returns:
            Total cost in USD.
        """

        def _query(session: Session) -> float:
            statement = select(func.sum(LLMCall.cost_usd))
            result = session.exec(statement).one_or_none()
            return float(result) if result else 0.0

        return await self._run_session(_query)

    async def get_topic_costs(self) -> dict[str, float]:
        """Get total spend grouped by topic.

        Returns:
            Dictionary mapping topic_slug to total USD spend.
        """

        def _query(session: Session) -> dict[str, float]:
            statement = (
                select(LLMCall.topic_slug, func.sum(LLMCall.cost_usd))
                .where(LLMCall.topic_slug.is_not(None))
                .group_by(LLMCall.topic_slug)
            )
            results = session.exec(statement).all()
            return {topic: float(cost) for topic, cost in results}

        return await self._run_session(_query)
