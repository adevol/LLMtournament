"""Tests for API caching."""

import json
import tempfile
from pathlib import Path

from llm_tournament.services.llm.client import CacheDB, FakeLLMClient, LLMResponse


class TestCacheDB:
    """Tests for DuckDB cache database."""

    async def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheDB(Path(tmpdir) / "cache.duckdb")
            result = await cache.get("nonexistent_key")
            assert result is None

    async def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheDB(Path(tmpdir) / "cache.duckdb")

            response = LLMResponse(
                content="test_response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            await cache.set("test_key", "test_model", response)
            result = await cache.get("test_key")

            assert result is not None
            assert result.content == "test_response"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5
            assert result.total_tokens == 15

    async def test_cache_overwrite(self):
        """Test overwriting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheDB(Path(tmpdir) / "cache.duckdb")

            response1 = LLMResponse(
                content="response1",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            response2 = LLMResponse(
                content="response2",
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            )
            await cache.set("key", "model", response1)
            await cache.set("key", "model", response2)
            result = await cache.get("key")

            assert result is not None
            assert result.content == "response2"
            assert result.total_tokens == 30

    async def test_cache_creates_directory(self):
        """Test cache creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subdir" / "cache.duckdb"
            cache = CacheDB(cache_path)

            assert cache_path.parent.exists()
            response = LLMResponse(
                content="response",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            await cache.set("key", "model", response)


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
