"""Tests for match engine helpers."""

import pytest

from llm_tournament.services.llm import FakeLLMClient
from llm_tournament.services.match.engine import (
    JudgeRotation,
    _parse_with_repair,
    _summarize_votes,
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
