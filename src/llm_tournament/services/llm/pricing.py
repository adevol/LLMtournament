"""OpenRouter model pricing service.

Fetches and caches model pricing from OpenRouter's /api/v1/models endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing info for a single model."""

    model_id: str
    prompt_price: float  # USD per token
    completion_price: float  # USD per token
    context_length: int

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute total cost for a request.

        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Total cost in USD.
        """
        return (prompt_tokens * self.prompt_price) + (completion_tokens * self.completion_price)


class PricingService:
    """Fetches and caches model pricing from OpenRouter.

    Usage:
        pricing = PricingService(api_key)
        await pricing.refresh()
        cost = pricing.compute_cost("gpt-4", prompt_tokens=100, completion_tokens=50)
    """

    def __init__(self, api_key: str) -> None:
        """Initialize pricing service.

        Args:
            api_key: OpenRouter API key.
        """
        self.api_key = api_key
        self._pricing: dict[str, ModelPricing] = {}
        self._client = httpx.AsyncClient(timeout=30.0)

    async def refresh(self) -> None:
        """Fetch latest pricing from OpenRouter API."""
        logger.info("pricing_refresh_start")

        response = await self._client.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()

        data = response.json()
        models: list[dict[str, Any]] = data.get("data", [])

        self._pricing.clear()
        for model in models:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})

            # Pricing values are strings like "0.000001" (USD per token)
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))
            context_length = model.get("context_length", 0)

            self._pricing[model_id] = ModelPricing(
                model_id=model_id,
                prompt_price=prompt_price,
                completion_price=completion_price,
                context_length=context_length,
            )

        logger.info("pricing_refresh_complete", models_loaded=len(self._pricing))

    def get_pricing(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a specific model.

        Args:
            model_id: OpenRouter model ID.

        Returns:
            ModelPricing or None if not found.
        """
        return self._pricing.get(model_id)

    def compute_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        """Compute cost for a model request.

        Args:
            model_id: OpenRouter model ID.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Cost in USD, or None if model pricing unknown.
        """
        pricing = self.get_pricing(model_id)
        if pricing is None:
            logger.warning("pricing_unknown", model_id=model_id)
            return None
        return pricing.compute_cost(prompt_tokens, completion_tokens)

    def list_models(self) -> list[str]:
        """List all available model IDs.

        Returns:
            List of model IDs with cached pricing.
        """
        return list(self._pricing.keys())

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> PricingService:
        """Async context manager entry."""
        await self.refresh()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()
