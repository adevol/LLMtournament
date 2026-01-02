"""Judge rotation and audit logic for LLM Tournament."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, ValidationError

from llm_tournament.prompts import (
    judge_strict_retry_prompt,
    judge_system_prompt,
    judge_user_prompt,
)
from llm_tournament.services.llm import LLMClient

logger = structlog.get_logger()


class JudgeResult(BaseModel):
    """Parsed judge response.

    Attributes:
        winner: "A" or "B".
        confidence: Confidence score 0.0-1.0.
        reasons: List of reasoning points.
        winner_edge: Why the winner beat the loser.
    """

    winner: str
    confidence: float
    reasons: list[str]
    winner_edge: str


@dataclass
class JudgeRotation:
    """Manages judge rotation and audit logic.

    Attributes:
        judge_models: List of judge model IDs.
        current_index: Current position in rotation.
    """

    judge_models: list[str]
    current_index: int = 0

    def next_judge(self) -> str:
        """Get next judge in rotation.

        Returns:
            Judge model ID.
        """
        judge = self.judge_models[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.judge_models)
        return judge

    def get_audit_judges(self, exclude: str) -> list[str]:
        """Get judges for audit (excluding primary).

        Args:
            exclude: Primary judge to exclude.

        Returns:
            List of available audit judges.
        """
        return [j for j in self.judge_models if j != exclude]


def parse_judge_response(response: str) -> JudgeResult:
    """Parse judge response JSON.

    Args:
        response: Raw response string (may contain markdown).

    Returns:
        Parsed JudgeResult.

    Raises:
        ValueError: If parsing fails.
    """
    json_text = response.strip()

    # Remove markdown code blocks if present
    if "```" in json_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_text)
        if match:
            json_text = match.group(1)

    # Try to find JSON object
    match = re.search(r"\{[\s\S]*\}", json_text)
    if match:
        json_text = match.group(0)

    try:
        data = json.loads(json_text)
        return JudgeResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        msg = f"Failed to parse judge response: {e}"
        raise ValueError(msg) from e


def repair_json(broken_json: str) -> str:
    """Attempt lightweight JSON repair.

    Args:
        broken_json: Potentially malformed JSON.

    Returns:
        Repaired JSON string.
    """
    # Common fixes
    text = broken_json.strip()

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([\}\]])", r"\1", text)

    # Ensure quotes around keys
    text = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1"\2":', text)

    # Fix single quotes to double quotes
    return text.replace("'", '"')


@dataclass
class MatchResult:
    """Result of a single match.

    Attributes:
        essay_a_id: ID of essay A.
        essay_b_id: ID of essay B.
        winner: "A" or "B".
        confidence: Judge confidence.
        reasons: List of reasons.
        winner_edge: Why winner won.
        primary_judge: Primary judge model.
        audit_judges: List of audit judges used (if any).
        final_decision: "primary", "unanimous", or "majority".
    """

    essay_a_id: str
    essay_b_id: str
    winner: str
    confidence: float
    reasons: list[str]
    winner_edge: str
    primary_judge: str
    audit_judges: list[str] = field(default_factory=list)
    final_decision: str = "primary"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


async def judge_match(
    client: LLMClient,
    essay_a: str,
    essay_b: str,
    _essay_a_id: str,
    _essay_b_id: str,
    judge_model: str,
    max_tokens: int,
    temperature: float,
    rotation: JudgeRotation | None = None,
    _attempted_judges: set[str] | None = None,
) -> JudgeResult:
    """Have a judge evaluate a pair of essays.

    Args:
        client: Async LLM client.
        essay_a: Full text of essay A.
        essay_b: Full text of essay B.
        essay_a_id: ID of essay A.
        essay_b_id: ID of essay B.
        judge_model: Judge model ID.
        max_tokens: Max tokens for response.
        temperature: Sampling temperature.
        rotation: Optional judge rotation for fallback on parse failure.
        _attempted_judges: Internal set of already-tried judges.

    Returns:
        Parsed JudgeResult.
    """
    attempted = _attempted_judges or {judge_model}

    messages = [
        {"role": "system", "content": judge_system_prompt()},
        {"role": "user", "content": judge_user_prompt(essay_a, essay_b)},
    ]

    response = await client.complete(judge_model, messages, max_tokens, temperature)

    try:
        return parse_judge_response(response)
    except ValueError:
        logger.warning("judge_parse_failed", judge=judge_model, retrying=True)

        messages = [
            {"role": "system", "content": judge_system_prompt()},
            {"role": "user", "content": judge_strict_retry_prompt(essay_a, essay_b)},
        ]
        response = await client.complete(judge_model, messages, max_tokens, temperature)

        try:
            return parse_judge_response(response)
        except ValueError:
            repaired = repair_json(response)
            try:
                return parse_judge_response(repaired)
            except ValueError:
                if rotation:
                    fallback_judges = [
                        j for j in rotation.judge_models if j not in attempted
                    ]
                    if fallback_judges:
                        fallback = fallback_judges[0]
                        logger.warning(
                            "judge_fallback",
                            failed_judge=judge_model,
                            fallback_judge=fallback,
                        )
                        attempted.add(fallback)
                        return await judge_match(
                            client,
                            essay_a,
                            essay_b,
                            _essay_a_id,
                            _essay_b_id,
                            fallback,
                            max_tokens,
                            temperature,
                            rotation,
                            attempted,
                        )
                raise


async def run_match_with_audit(
    client: LLMClient,
    essay_a: str,
    essay_b: str,
    essay_a_id: str,
    essay_b_id: str,
    rotation: JudgeRotation,
    audit_threshold: float,
    max_tokens: int,
    temperature: float,
) -> MatchResult:
    """Run a match with potential audit judges.

    Args:
        client: Async LLM client.
        essay_a: Essay A text.
        essay_b: Essay B text.
        essay_a_id: Essay A identifier.
        essay_b_id: Essay B identifier.
        rotation: Judge rotation manager.
        audit_threshold: Confidence threshold for audit.
        max_tokens: Max tokens.
        temperature: Temperature.

    Returns:
        Final MatchResult.
    """
    # Primary judgment
    primary_judge = rotation.next_judge()
    primary_result = await judge_match(
        client,
        essay_a,
        essay_b,
        essay_a_id,
        essay_b_id,
        primary_judge,
        max_tokens,
        temperature,
        rotation,
    )

    logger.info(
        "primary_judgment",
        judge=primary_judge,
        winner=primary_result.winner,
        confidence=primary_result.confidence,
    )

    if primary_result.confidence >= audit_threshold:
        return MatchResult(
            essay_a_id=essay_a_id,
            essay_b_id=essay_b_id,
            winner=primary_result.winner,
            confidence=primary_result.confidence,
            reasons=primary_result.reasons,
            winner_edge=primary_result.winner_edge,
            primary_judge=primary_judge,
            final_decision="primary",
        )

    audit_judges = rotation.get_audit_judges(primary_judge)
    if not audit_judges:
        return MatchResult(
            essay_a_id=essay_a_id,
            essay_b_id=essay_b_id,
            winner=primary_result.winner,
            confidence=primary_result.confidence,
            reasons=primary_result.reasons,
            winner_edge=primary_result.winner_edge,
            primary_judge=primary_judge,
            final_decision="primary",
        )

    second_judge = audit_judges[0]
    second_result = await judge_match(
        client,
        essay_a,
        essay_b,
        essay_a_id,
        essay_b_id,
        second_judge,
        max_tokens,
        temperature,
        rotation,
    )

    logger.info(
        "audit_judgment",
        judge=second_judge,
        winner=second_result.winner,
        confidence=second_result.confidence,
    )

    # Check agreement
    if primary_result.winner == second_result.winner:
        # Unanimous
        avg_confidence = (primary_result.confidence + second_result.confidence) / 2
        return MatchResult(
            essay_a_id=essay_a_id,
            essay_b_id=essay_b_id,
            winner=primary_result.winner,
            confidence=avg_confidence,
            reasons=primary_result.reasons + second_result.reasons,
            winner_edge=primary_result.winner_edge,
            primary_judge=primary_judge,
            audit_judges=[second_judge],
            final_decision="unanimous",
        )

    min_audits_for_tiebreak = 2
    if len(audit_judges) < min_audits_for_tiebreak:
        if primary_result.confidence >= second_result.confidence:
            winner_result = primary_result
            winner_judge = primary_judge
        else:
            winner_result = second_result
            winner_judge = second_judge

        return MatchResult(
            essay_a_id=essay_a_id,
            essay_b_id=essay_b_id,
            winner=winner_result.winner,
            confidence=winner_result.confidence,
            reasons=winner_result.reasons,
            winner_edge=winner_result.winner_edge,
            primary_judge=winner_judge,
            audit_judges=(
                [second_judge] if winner_judge == primary_judge else [primary_judge]
            ),
            final_decision="confidence_tiebreak",
        )

    third_judge = audit_judges[1]
    third_result = await judge_match(
        client,
        essay_a,
        essay_b,
        essay_a_id,
        essay_b_id,
        third_judge,
        max_tokens,
        temperature,
        rotation,
    )

    logger.info(
        "tiebreak_judgment",
        judge=third_judge,
        winner=third_result.winner,
        confidence=third_result.confidence,
    )

    votes = [primary_result.winner, second_result.winner, third_result.winner]
    winner = "A" if votes.count("A") > votes.count("B") else "B"
    if winner == primary_result.winner:
        base_result = primary_result
    elif winner == second_result.winner:
        base_result = second_result
    else:
        base_result = third_result

    avg_confidence = (
        primary_result.confidence + second_result.confidence + third_result.confidence
    ) / 3

    return MatchResult(
        essay_a_id=essay_a_id,
        essay_b_id=essay_b_id,
        winner=winner,
        confidence=avg_confidence,
        reasons=base_result.reasons,
        winner_edge=base_result.winner_edge,
        primary_judge=primary_judge,
        audit_judges=[second_judge, third_judge],
        final_decision="majority",
    )
