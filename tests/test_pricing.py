"""Tests for PricingService."""

import pytest

from llm_tournament.services.llm.pricing import ModelPricing, PricingService


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_compute_cost_basic(self) -> None:
        """Test basic cost computation."""
        pricing = ModelPricing(
            model_id="test-model",
            prompt_price=0.00001,
            completion_price=0.00003,
            context_length=4096,
        )
        # 100 prompt tokens * 0.00001 + 50 completion tokens * 0.00003
        # = 0.001 + 0.0015 = 0.0025
        cost = pricing.compute_cost(prompt_tokens=100, completion_tokens=50)
        assert cost == pytest.approx(0.0025)

    def test_compute_cost_zero_tokens(self) -> None:
        """Test cost with zero tokens."""
        pricing = ModelPricing(
            model_id="test-model",
            prompt_price=0.00001,
            completion_price=0.00003,
            context_length=4096,
        )
        cost = pricing.compute_cost(prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_compute_cost_free_model(self) -> None:
        """Test cost for free model (zero pricing)."""
        pricing = ModelPricing(
            model_id="free-model",
            prompt_price=0.0,
            completion_price=0.0,
            context_length=8192,
        )
        cost = pricing.compute_cost(prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0


class TestPricingService:
    """Tests for PricingService."""

    def test_compute_cost_unknown_model(self) -> None:
        """Test compute_cost returns None for unknown model."""
        service = PricingService(api_key="test-key")
        cost = service.compute_cost("unknown-model", 100, 50)
        assert cost is None

    def test_list_models_empty(self) -> None:
        """Test list_models returns empty before refresh."""
        service = PricingService(api_key="test-key")
        assert service.list_models() == []

    def test_get_pricing_unknown(self) -> None:
        """Test get_pricing returns None for unknown model."""
        service = PricingService(api_key="test-key")
        assert service.get_pricing("unknown-model") is None
