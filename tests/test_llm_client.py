"""Tests for fake LLM client behavior."""

import json
import random
from unittest.mock import AsyncMock

import httpx
import pytest

from llm_tournament.services.llm.client import (
    FakeLLMClient,
    LLMResponse,
    OpenRouterClient,
    _IncompleteResponseError,
)


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

    async def test_complete_prompt_builds_standard_messages(self):
        """complete_prompt should behave like a direct system+user complete call."""
        model = "test"
        system_prompt = "You are a pairwise judge."
        user_prompt = "Compare Essay A and Essay B and return winner in JSON."
        max_tokens = 100
        temperature = 0.7

        direct_client = FakeLLMClient(seed=42)
        helper_client = FakeLLMClient(seed=42)

        direct = await direct_client.complete(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens,
            temperature,
        )
        via_helper = await helper_client.complete_prompt(
            model,
            system_prompt,
            user_prompt,
            max_tokens,
            temperature,
        )

        assert direct == via_helper


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
        """Raises request error after exhausting retries."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        post_mock = AsyncMock(
            side_effect=httpx.ConnectTimeout("timeout", request=request),
        )
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            with pytest.raises(httpx.ConnectTimeout):
                await client.complete(
                    "test/model",
                    [{"role": "user", "content": "hello"}],
                    100,
                    0.7,
                )
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

    async def test_openrouter_http_status_error_exhausts_after_three_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Raises HTTP status error after exhausting retries."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        post_mock = AsyncMock(
            side_effect=[
                httpx.Response(500, request=request, json={"error": "server error"}),
                httpx.Response(500, request=request, json={"error": "server error"}),
                httpx.Response(500, request=request, json={"error": "server error"}),
            ]
        )
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.complete(
                    "test/model",
                    [{"role": "user", "content": "hello"}],
                    100,
                    0.7,
                )
            assert post_mock.await_count == 3
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

    async def test_openrouter_retries_on_incomplete_response_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Retries blank completions and returns first non-empty completion."""
        client = OpenRouterClient(api_key="test-key")
        incomplete = LLMResponse(
            content="   ",
            prompt_tokens=1,
            completion_tokens=0,
            total_tokens=1,
        )
        complete = LLMResponse(
            content="ok",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        call_api_mock = AsyncMock(side_effect=[incomplete, complete])
        monkeypatch.setattr(client, "_call_api", call_api_mock)

        try:
            result = await client.complete(
                "test/model",
                [{"role": "user", "content": "hello"}],
                100,
                0.7,
            )
            assert result.content == "ok"
            assert call_api_mock.await_count == 2
        finally:
            await client.close()

    async def test_openrouter_incomplete_response_exhausts_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Raises when all attempts return empty content."""
        client = OpenRouterClient(api_key="test-key")
        incomplete = LLMResponse(
            content="",
            prompt_tokens=1,
            completion_tokens=0,
            total_tokens=1,
        )
        call_api_mock = AsyncMock(side_effect=[incomplete, incomplete, incomplete])
        monkeypatch.setattr(client, "_call_api", call_api_mock)

        try:
            with pytest.raises(_IncompleteResponseError):
                await client.complete(
                    "test/model",
                    [{"role": "user", "content": "hello"}],
                    100,
                    0.7,
                )
            assert call_api_mock.await_count == 3
        finally:
            await client.close()

    async def test_openrouter_raises_clear_error_on_malformed_response(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Malformed successful responses raise clear parse errors."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        malformed = httpx.Response(
            200,
            request=request,
            json={"choices": []},
        )
        post_mock = AsyncMock(return_value=malformed)
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            with pytest.raises(
                ValueError,
                match=r"Malformed OpenRouter response: missing choices\[0\]\.message\.content",
            ):
                await client.complete(
                    "test/model",
                    [{"role": "user", "content": "hello"}],
                    100,
                    0.7,
                )
            assert post_mock.await_count == 1
        finally:
            await client.close()

    async def test_openrouter_handles_non_numeric_usage_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-numeric usage fields are safely coerced to zero."""
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request("POST", OpenRouterClient.BASE_URL)
        response_ok = httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": "not-a-number",
                    "completion_tokens": None,
                    "total_tokens": "3.14",
                },
            },
        )
        post_mock = AsyncMock(return_value=response_ok)
        monkeypatch.setattr(client.client, "post", post_mock)

        try:
            result = await client.complete(
                "test/model",
                [{"role": "user", "content": "hello"}],
                100,
                0.7,
            )
            assert result.content == "ok"
            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.total_tokens == 0
        finally:
            await client.close()
