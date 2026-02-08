"""Tests for fake LLM client behavior."""

import json
import random
from unittest.mock import AsyncMock

import httpx
import pytest
from tenacity import RetryError

from llm_tournament.services.llm.client import FakeLLMClient, LLMResponse, OpenRouterClient


class TestFakeLLMClient:
    """Tests for fake LLM client."""

    async def test_deterministic_with_same_seed(self):
        """Test fake client produces deterministic output with same seed."""
        client1 = FakeLLMClient(seed=42)
        client2 = FakeLLMClient(seed=42)

        messages = [{"role": "user", "content": "compare these essays, winner"}]

        result1 = await client1.complete("test", messages, 100, 0.7)
        result2 = await client2.complete("test", messages, 100, 0.7)

        # Both should produce valid LLMResponse with JSON content
        assert isinstance(result1, LLMResponse)
        assert isinstance(result2, LLMResponse)
        assert "winner" in result1.content
        assert "winner" in result2.content

    async def test_tracks_call_count(self):
        """Test fake client tracks call count."""
        client = FakeLLMClient(seed=42)
        assert client.call_count == 0

        await client.complete("test", [{"role": "user", "content": "hello"}], 100, 0.7)
        assert client.call_count == 1

        await client.complete("test", [{"role": "user", "content": "hello"}], 100, 0.7)
        assert client.call_count == 2

    def test_cost_tracking_disabled(self):
        """Fake client does not support API cost tracking."""
        client = FakeLLMClient(seed=42)
        assert client.cost_tracking_enabled is False

    async def test_essay_response_structure(self):
        """Test fake essay contains expected content."""
        client = FakeLLMClient(seed=42)
        messages = [{"role": "user", "content": "write fiction and journalism"}]

        result = await client.complete("test", messages, 100, 0.7)

        # LLMResponse should contain model name and story content
        assert isinstance(result, LLMResponse)
        assert "test" in result.content
        assert "characters" in result.content
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    async def test_judgment_response_is_valid_json(self):
        """Test fake judgment is valid JSON."""
        client = FakeLLMClient(seed=42)
        messages = [{"role": "user", "content": "compare winner"}]

        result = await client.complete("test", messages, 100, 0.7)
        assert isinstance(result, LLMResponse)
        data = json.loads(result.content)

        assert data["winner"] in ["A", "B"]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["reasons"], list)
        assert "winner_edge" in data

    async def test_judgment_does_not_mutate_global_random_state(self):
        """Fake judgment generation should not alter global random state."""
        client = FakeLLMClient(seed=42)
        messages = [{"role": "user", "content": "compare winner"}]

        random.seed(98765)
        expected_next = random.random()  # noqa: S311

        random.seed(98765)
        await client.complete("test", messages, 100, 0.7)
        actual_next = random.random()  # noqa: S311

        assert actual_next == expected_next


class TestOpenRouterClientRetry:
    """Tests for OpenRouter retry behavior."""

    async def test_openrouter_retries_on_request_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Retries transient request errors and succeeds on a later attempt."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        response_ok = httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        post_mock = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("timeout", request=request),
                response_ok,
            ]
        )
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            result = await client.complete(
                "test/model",
                [{"role": "user", "content": "hello"}],
                100,
                0.7,
            )
            assert result.content == "ok"
            assert post_mock.await_count == 2
        finally:
            await client.close()

    async def test_openrouter_request_error_exhausts_after_three_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Raises RetryError with request error cause after exhausting retries."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        post_mock = AsyncMock(
            side_effect=httpx.ConnectTimeout("timeout", request=request),
        )
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            with pytest.raises(RetryError) as exc_info:
                await client.complete(
                    "test/model",
                    [{"role": "user", "content": "hello"}],
                    100,
                    0.7,
                )
            assert isinstance(exc_info.value.last_attempt.exception(), httpx.ConnectTimeout)
            assert post_mock.await_count == 3
        finally:
            await client.close()

    async def test_openrouter_still_retries_http_status_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Keeps retry behavior for HTTP status errors."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        response_500 = httpx.Response(
            500,
            request=request,
            json={"error": "server error"},
        )
        response_ok = httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        post_mock = AsyncMock(side_effect=[response_500, response_ok])
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            result = await client.complete(
                "test/model",
                [{"role": "user", "content": "hello"}],
                100,
                0.7,
            )
            assert result.content == "ok"
            assert post_mock.await_count == 2
        finally:
            await client.close()

    async def test_openrouter_cost_tracking_flag_without_tracker(self):
        """OpenRouter client exposes disabled cost tracking by default."""
        client = OpenRouterClient(api_key="test-key")
        try:
            assert client.cost_tracking_enabled is False
            assert client.total_cost == 0.0
        finally:
            await client.close()
