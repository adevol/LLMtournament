"""Tests for match engine helpers."""

import pytest

from llm_tournament.services.llm import FakeLLMClient
from llm_tournament.services.llm.client import IncompleteResponseError, LLMClient, LLMResponse
from llm_tournament.services.match.engine import (
    JudgeRotation,
    _parse_with_repair,
    _summarize_votes,
    run_match_parallel_majority,
    run_match_with_audit,
)


def test_parse_with_repair_handles_broken_json():
    """Parses and repairs common malformed JSON responses."""
    response = "{winner: 'A', confidence: 0.8, reasons: ['ok'], winner_edge: 'edge',}"
    result = _parse_with_repair(response)

    assert result.winner == "A"
    assert result.confidence == 0.8
    assert result.reasons == ["ok"]
    assert result.winner_edge == "edge"


def test_summarize_votes_requires_results():
    """Empty vote sets should raise a clear error."""
    with pytest.raises(ValueError, match="At least one judge result is required"):
        _summarize_votes([])


@pytest.mark.asyncio
async def test_run_match_with_audit_requires_judges():
    """Audit matches require at least one judge."""
    rotation = JudgeRotation([])
    client = FakeLLMClient()

    with pytest.raises(ValueError, match="At least one judge is required"):
        await run_match_with_audit(
            client,
            essay_a="A",
            essay_b="B",
            essay_a_id="a",
            essay_b_id="b",
            rotation=rotation,
            audit_threshold=0.7,
            max_tokens=10,
            temperature=0.2,
        )


class _FlakyJudgeClient(LLMClient):
    @property
    def total_cost(self) -> float:
        return 0.0

    async def complete(
        self,
        model: str,
        _messages,
        _max_tokens: int,
        _temperature: float,
    ) -> LLMResponse:
        if model == "bad/judge":
            raise IncompleteResponseError()
        return LLMResponse(
            content='{"winner":"A","confidence":0.8,"reasons":["clear"],"winner_edge":"better"}',
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


@pytest.mark.asyncio
async def test_parallel_majority_tolerates_single_judge_incomplete_response():
    client = _FlakyJudgeClient()

    result = await run_match_parallel_majority(
        client=client,
        essay_a="Essay A",
        essay_b="Essay B",
        essay_a_id="a",
        essay_b_id="b",
        primary_judges=["bad/judge", "good/judge-1", "good/judge-2"],
        sub_judges=[],
        confidence_threshold=0.7,
        max_tokens=100,
        temperature=0.2,
    )

    assert result.winner == "A"
    assert result.final_decision == "parallel_majority_3"
