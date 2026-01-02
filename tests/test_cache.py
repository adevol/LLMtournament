"""Tests for API caching."""

import json
import tempfile
from pathlib import Path

from llm_tournament.services.llm.client import CacheDB, FakeLLMClient


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

            await cache.set("test_key", "test_model", "test_response")
            result = await cache.get("test_key")

            assert result == "test_response"

    async def test_cache_overwrite(self):
        """Test overwriting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheDB(Path(tmpdir) / "cache.duckdb")

            await cache.set("key", "model", "response1")
            await cache.set("key", "model", "response2")
            result = await cache.get("key")

            assert result == "response2"

    async def test_cache_creates_directory(self):
        """Test cache creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subdir" / "cache.duckdb"
            cache = CacheDB(cache_path)

            assert cache_path.parent.exists()
            await cache.set("key", "model", "response")


class TestFakeLLMClient:
    """Tests for fake LLM client."""

    async def test_deterministic_with_same_seed(self):
        """Test fake client produces deterministic output with same seed."""
        client1 = FakeLLMClient(seed=42)
        client2 = FakeLLMClient(seed=42)

        messages = [{"role": "user", "content": "compare these essays, winner"}]

        result1 = await client1.complete("test", messages, 100, 0.7)
        result2 = await client2.complete("test", messages, 100, 0.7)

        # Both should produce valid JSON with same structure
        assert "winner" in result1
        assert "winner" in result2

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

        # Simple essay response contains model name and story content
        assert "test" in result
        assert "characters" in result

    async def test_judgment_response_is_valid_json(self):
        """Test fake judgment is valid JSON."""
        client = FakeLLMClient(seed=42)
        messages = [{"role": "user", "content": "compare winner"}]

        result = await client.complete("test", messages, 100, 0.7)
        data = json.loads(result)

        assert data["winner"] in ["A", "B"]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["reasons"], list)
        assert "winner_edge" in data
