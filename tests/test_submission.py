"""Tests for submission service fallback behavior."""

import asyncio

import pytest

from llm_tournament.core.config import TopicConfig, TournamentConfig
from llm_tournament.services.llm import IncompleteResponseError, LLMClient
from llm_tournament.services.submission import SubmissionService


class _AlwaysIncompleteClient(LLMClient):
    @property
    def total_cost(self) -> float:
        return 0.0

    async def complete(self, _model: str, _messages, _max_tokens: int, _temperature: float):
        raise IncompleteResponseError()


class _SubmissionStoreStub:
    def __init__(self) -> None:
        self.feedback_saved: list[str] = []
        self.revision_saved: list[str] = []

    async def load_essay(self, _topic_slug: str, _essay_id: str, _version: str) -> str:
        return "Original essay body."

    async def save_feedback(
        self, _topic_slug: str, _writer_slug: str, _critic_slug: str, content: str
    ) -> None:
        self.feedback_saved.append(content)

    async def load_feedback(self, _topic_slug: str, _writer_slug: str, _critic_slug: str) -> str:
        return "Feedback text."

    async def save_revision(
        self, _topic_slug: str, _writer_slug: str, _critic_slug: str, content: str
    ) -> None:
        self.revision_saved.append(content)


@pytest.fixture
def config() -> TournamentConfig:
    return TournamentConfig(
        writers=["test/writer-a"],
        critics=["test/critic-a"],
        judges=["test/judge-a"],
        topics=[TopicConfig(title="Test Topic", prompts={"Essay": "Write an essay"})],
        simple_mode=False,
    )


@pytest.mark.asyncio
async def test_critique_falls_back_on_incomplete_response(config: TournamentConfig) -> None:
    store = _SubmissionStoreStub()
    service = SubmissionService(
        config,
        _AlwaysIncompleteClient(),
        store,  # type: ignore[arg-type]
        asyncio.Semaphore(1),
    )

    topic = config.topics[0]
    await service.run_critique_batch(topic, config.get_writer_specs(), config.get_critic_specs())

    assert len(store.feedback_saved) == 1
    assert "incomplete" in store.feedback_saved[0].lower()


@pytest.mark.asyncio
async def test_revision_falls_back_to_original_essay(config: TournamentConfig) -> None:
    store = _SubmissionStoreStub()
    service = SubmissionService(
        config,
        _AlwaysIncompleteClient(),
        store,  # type: ignore[arg-type]
        asyncio.Semaphore(1),
    )

    topic = config.topics[0]
    await service.run_revision_batch(topic, config.get_writer_specs(), config.get_critic_specs())

    assert store.revision_saved == ["Original essay body."]
