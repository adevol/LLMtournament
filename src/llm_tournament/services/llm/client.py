"""OpenRouter API client with async support and retries."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

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

    @property
    @abstractmethod
    def total_cost(self) -> float:
        """Return total cost of all API calls made by this client."""

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
        self._rng = random.Random(seed)  # noqa: S311

    @property
    def total_cost(self) -> float:
        """Fake client has zero cost."""
        return 0.0

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
        winner = self._rng.choice(["A", "B"])
        confidence = round(self._rng.uniform(0.6, 0.95), 2)
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


class _IncompleteResponseError(RuntimeError):
    """Raised when the model returns an incomplete response."""


class OpenRouterClient(LLMClient):
    """Async OpenRouter API client with retries."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    _INCOMPLETE_RETRIES = 2

    def __init__(
        self,
        api_key: str,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            cost_tracker: Optional cost tracker for recording API costs.
        """
        self.api_key = api_key
        self.cost_tracker = cost_tracker
        self.client = httpx.AsyncClient(timeout=120.0)
        self._total_cost = 0.0

    @property
    def total_cost(self) -> float:
        """Return total accumulated cost."""
        return self._total_cost

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
        response = await self._call_with_retries(
            model,
            messages,
            max_tokens,
            temperature,
        )
        await self._record_cost(model, response)
        return response

    async def _record_cost(self, model: str, response: LLMResponse) -> None:
        if not self.cost_tracker:
            return
        call_cost = await self.cost_tracker.record_call(
            model,
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            role="api",
        )
        self._total_cost += call_cost

    async def _call_with_retries(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        response: LLMResponse | None = None

        async def _attempt_call() -> LLMResponse:
            nonlocal response
            response = await self._call_api(model, messages, max_tokens, temperature)
            if not response.content.strip():
                raise _IncompleteResponseError()
            return response

        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_IncompleteResponseError),
            wait=wait_fixed(0),
            stop=stop_after_attempt(self._INCOMPLETE_RETRIES + 1),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await _attempt_call()

        if response is None:
            raise _IncompleteResponseError()
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
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
            httpx.HTTPStatusError: On HTTP API error after retries.
            httpx.RequestError: On network/transport error after retries.
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
    dry_run: bool = False,
    seed: int = 42,
    cost_tracker: CostTracker | None = None,
) -> LLMClient:
    """Create appropriate LLM client based on settings.

    Args:
        api_key: OpenRouter API key (required unless dry_run).
        dry_run: If True, use fake client.
        seed: Random seed for fake client.
        cost_tracker: Optional cost tracker.

    Returns:
        LLMClient instance.
    """
    if dry_run:
        logger.info("using_fake_client", seed=seed)
        return FakeLLMClient(seed=seed)

    if not api_key:
        msg = "API key required for real API calls"
        raise ValueError(msg)

    return OpenRouterClient(
        api_key=api_key,
        cost_tracker=cost_tracker,
    )
