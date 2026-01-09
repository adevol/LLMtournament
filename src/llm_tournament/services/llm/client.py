"""OpenRouter API client with async support, caching and retries."""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import httpx
import structlog
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_tournament.core.config import hash_messages
from llm_tournament.services.llm.cost_tracker import CostTracker

logger = structlog.get_logger()

_FAKE_RESPONSES_PATH = Path(__file__).parent / "fake_responses.yaml"


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM API call with usage data."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _load_fake_responses() -> dict[str, Any]:
    """Load fake response templates from YAML file (cached after first call).

    Returns:
        Dictionary containing response templates.
    """
    if not hasattr(_load_fake_responses, "_cache"):
        with _FAKE_RESPONSES_PATH.open(encoding="utf-8") as f:
            _load_fake_responses._cache = yaml.safe_load(f)
    return _load_fake_responses._cache


class LLMClient(ABC):
    """Abstract base class for async LLM clients."""

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate a completion from the model.

        Args:
            model: Model identifier.
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and usage data.
        """

    async def close(self) -> None:  # noqa: B027
        """Close any resources. Override if needed."""


class FakeLLMClient(LLMClient):
    """Fake async LLM client for testing and dry runs."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize fake client with seed.

        Args:
            seed: Random seed for deterministic responses.
        """
        self.seed = seed
        self.call_count = 0

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        _max_tokens: int,
        _temperature: float,
    ) -> LLMResponse:
        """Return deterministic fake response.

        Args:
            model: Model identifier (used in response).
            messages: Input messages (last one used for context).
            _max_tokens: Maximum tokens (unused).
            _temperature: Temperature (unused).

        Returns:
            LLMResponse with fake content and simulated token counts.
        """
        self.call_count += 1
        last_content = messages[-1]["content"] if messages else ""
        last_lower = last_content.lower()

        system_content = ""
        if messages and messages[0].get("role") == "system":
            system_content = messages[0].get("content", "").lower()

        content = None
        if (
            ("essay a" in last_lower and "essay b" in last_lower)
            or ("winner" in last_lower and "json" in last_lower)
            or ("compare" in last_lower and ("essay" in last_lower or "winner" in last_lower))
            or ("pairwise" in system_content)
            or ("judge" in system_content)
        ):
            content = self._fake_judgment()
        elif "feedback" in last_lower or "critique" in last_lower:
            content = self._fake_feedback(model)
        elif "revise" in last_lower:
            content = self._fake_revision(model)
        else:
            content = self._fake_essay(model)

        # Simulate token counts based on content length
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 2
        completion_tokens = len(content.split()) * 2
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _fake_essay(self, model: str) -> str:
        """Generate a fake essay."""
        templates = _load_fake_responses()
        return templates["essay"].format(model=model)

    def _fake_feedback(self, model: str) -> str:
        """Generate fake critique feedback."""
        templates = _load_fake_responses()
        return templates["feedback"].format(model=model)

    def _fake_judgment(self) -> str:
        """Generate fake judgment JSON."""
        templates = _load_fake_responses()
        random.seed(self.seed + self.call_count)
        winner = random.choice(["A", "B"])  # noqa: S311
        confidence = round(random.uniform(0.6, 0.95), 2)  # noqa: S311
        judgment = templates["judgment"]
        return json.dumps(
            {
                "winner": winner,
                "confidence": confidence,
                "reasons": judgment["reasons"],
                "winner_edge": judgment["winner_edge"].format(winner=winner),
            }
        )

    def _fake_revision(self, model: str) -> str:
        """Generate a fake revised essay."""
        templates = _load_fake_responses()
        return templates["revision"].format(model=model)


class CacheDB:
    """DuckDB cache for API responses with async support."""

    def __init__(self, db_path: Path) -> None:
        """Initialize cache database.

        Args:
            db_path: Path to DuckDB database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key VARCHAR PRIMARY KEY,
                    model VARCHAR NOT NULL,
                    response VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        finally:
            conn.close()

    async def get(self, key: str) -> LLMResponse | None:
        """Get cached response (async).

        Args:
            key: Cache key (hash of messages + params).

        Returns:
            Cached LLMResponse or None if not found.
        """

        def _get() -> LLMResponse | None:
            conn = duckdb.connect(str(self.db_path))
            try:
                result = conn.execute(
                    "SELECT response FROM cache WHERE cache_key = ?", [key]
                ).fetchone()
                if result:
                    data = json.loads(result[0])
                    return LLMResponse(**data)
                return None
            finally:
                conn.close()

        return await asyncio.to_thread(_get)

    async def set(self, key: str, model: str, response: LLMResponse) -> None:
        """Store response in cache (async).

        Args:
            key: Cache key.
            model: Model identifier.
            response: LLMResponse to cache.
        """

        def _set() -> None:
            conn = duckdb.connect(str(self.db_path))
            try:
                response_json = json.dumps(
                    {
                        "content": response.content,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                    }
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (cache_key, model, response)
                    VALUES (?, ?, ?)
                    """,
                    [key, model, response_json],
                )
            finally:
                conn.close()

        await asyncio.to_thread(_set)


class OpenRouterClient(LLMClient):
    """Async OpenRouter API client with caching and retries."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        cache_db: CacheDB | None = None,
        use_cache: bool = True,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            cache_db: Optional cache database.
            use_cache: Whether to use caching.
            cost_tracker: Optional cost tracker for recording API costs.
        """
        self.api_key = api_key
        self.cache_db = cache_db
        self.use_cache = use_cache and cache_db is not None
        self.cost_tracker = cost_tracker
        self.client = httpx.AsyncClient(timeout=120.0)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate completion via OpenRouter API.

        Args:
            model: OpenRouter model ID.
            messages: Chat messages.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and usage data.
        """
        params = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        cache_key = hash_messages(messages, params)

        # Check cache
        if self.use_cache and self.cache_db:
            cached = await self.cache_db.get(cache_key)
            if cached is not None:
                logger.debug("cache_hit", model=model, key=cache_key[:8])
                return cached

        # Make API call
        response = await self._call_api(model, messages, max_tokens, temperature)

        # Record cost if tracker configured
        if self.cost_tracker:
            await self.cost_tracker.record_call(
                model,
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
                role="api",
            )

        # Store in cache
        if self.use_cache and self.cache_db:
            await self.cache_db.set(cache_key, model, response)

        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _call_api(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make API call with retries.

        Args:
            model: Model ID.
            messages: Chat messages.
            max_tokens: Max tokens.
            temperature: Temperature.

        Returns:
            LLMResponse with content and usage data.

        Raises:
            httpx.HTTPStatusError: On API error after retries.
        """
        logger.info("api_call", model=model, max_tokens=max_tokens)

        response = await self.client.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "LLM Tournament",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        response.raise_for_status()

        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        logger.debug(
            "api_response",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


def create_client(
    api_key: str | None = None,
    cache_path: Path | None = None,
    use_cache: bool = True,
    dry_run: bool = False,
    seed: int = 42,
) -> LLMClient:
    """Create appropriate LLM client based on settings.

    Args:
        api_key: OpenRouter API key (required unless dry_run).
        cache_path: Path to cache database.
        use_cache: Whether to use caching.
        dry_run: Use fake client instead of real API.
        seed: Random seed for fake client.

    Returns:
        LLMClient instance.
    """
    if dry_run:
        logger.info("using_fake_client", seed=seed)
        return FakeLLMClient(seed=seed)

    if not api_key:
        msg = "API key required for real API calls"
        raise ValueError(msg)

    cache_db = CacheDB(cache_path) if cache_path else None
    return OpenRouterClient(api_key, cache_db, use_cache)
