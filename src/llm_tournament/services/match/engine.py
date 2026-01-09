"""Judge rotation and match evaluation for LLM Tournament.

This module provides two judging methods:

1. **Audit mode** (`run_match_with_audit`):
   - Single primary judge evaluates the match
   - If confidence < threshold, a second judge is called
   - On disagreement, a third tiebreaker judge decides

2. **Parallel majority mode** (`run_match_parallel_majority`):
   - 3 judges evaluate in parallel
   - Winner determined by majority vote (2/3)
   - If avg confidence < threshold, 2 sub-judges are added (5 total, 3/5 vote)
"""

from __future__ import annotations

import asyncio
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

# Constants for parallel majority voting
PRIMARY_JUDGE_COUNT = 3
SUB_JUDGE_COUNT = 2


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


def _build_judge_messages(context: MatchContext, strict: bool) -> list[dict[str, str]]:
    """Build judge messages for a judgment request.

    Args:
        context: MatchContext with essay text.
        strict: Whether to use strict retry prompt.

    Returns:
        List of message dicts.
    """
    prompt = judge_strict_retry_prompt if strict else judge_user_prompt
    return [
        {"role": "system", "content": judge_system_prompt()},
        {"role": "user", "content": prompt(context.essay_a, context.essay_b)},
    ]


async def _request_judgment(
    client: LLMClient,
    judge_model: str,
    context: MatchContext,
    strict: bool,
) -> str:
    """Request a judgment response from the model.

    Args:
        client: Async LLM client.
        judge_model: Judge model ID.
        context: MatchContext with request settings.
        strict: Whether to use strict retry prompt.

    Returns:
        Raw response string from the model.
    """
    messages = _build_judge_messages(context, strict)
    response = await client.complete(judge_model, messages, context.max_tokens, context.temperature)
    return response.content


def _parse_with_repair(response: str) -> JudgeResult:
    """Parse a judge response, applying repair when needed.

    Args:
        response: Raw response string.

    Returns:
        Parsed JudgeResult.

    Raises:
        ValueError: If parsing fails after repair.
    """
    try:
        return parse_judge_response(response)
    except ValueError:
        repaired = repair_json(response)
        return parse_judge_response(repaired)


def _next_fallback_judge(rotation: JudgeRotation | None, attempted: set[str]) -> str | None:
    """Select the next fallback judge not yet attempted.

    Args:
        rotation: Judge rotation manager, if available.
        attempted: Judges already attempted.

    Returns:
        Fallback judge ID or None.
    """
    if not rotation:
        return None
    for judge in rotation.judge_models:
        if judge not in attempted:
            return judge
    return None


@dataclass(frozen=True)
class MatchContext:
    """Context for match evaluation.

    Attributes:
        essay_a_id: ID of essay A.
        essay_b_id: ID of essay B.
        essay_a: Essay A text.
        essay_b: Essay B text.
        max_tokens: Max tokens for judge responses.
        temperature: Sampling temperature.
        audit_threshold: Confidence threshold for audit/expansion.
    """

    essay_a_id: str
    essay_b_id: str
    essay_a: str
    essay_b: str
    max_tokens: int
    temperature: float
    audit_threshold: float | None = None


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


def _build_match_result(
    context: MatchContext,
    winner: str,
    confidence: float,
    reasons: list[str],
    winner_edge: str,
    primary_judge: str,
    audit_judges: list[str] | None = None,
    final_decision: str = "primary",
) -> MatchResult:
    """Build a MatchResult with consistent defaults.

    Args:
        context: MatchContext with essay IDs.
        winner: Winner label ("A" or "B").
        confidence: Confidence score.
        reasons: List of reasons.
        winner_edge: Winner edge explanation.
        primary_judge: Primary judge model ID.
        audit_judges: Optional list of audit judges.
        final_decision: Decision label.

    Returns:
        MatchResult instance.
    """
    return MatchResult(
        essay_a_id=context.essay_a_id,
        essay_b_id=context.essay_b_id,
        winner=winner,
        confidence=confidence,
        reasons=reasons,
        winner_edge=winner_edge,
        primary_judge=primary_judge,
        audit_judges=audit_judges or [],
        final_decision=final_decision,
    )


def _average_confidence(results: list[JudgeResult]) -> float:
    """Compute average confidence across judge results.

    Args:
        results: List of JudgeResult entries.

    Returns:
        Average confidence value.
    """
    return sum(r.confidence for r in results) / len(results)


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0] to keep math stable."""
    return max(0.0, min(1.0, value))


def _aggregate_confidence(results: list[JudgeResult], winner: str) -> float:
    """Aggregate confidence as support for the chosen winner.

    Converts each judge vote into a probability for the winner:
    - If judge voted for the winner, use confidence.
    - If judge voted against, use (1 - confidence).
    """
    supports = []
    for r in results:
        conf = _clamp_confidence(r.confidence)
        supports.append(conf if r.winner == winner else 1.0 - conf)
    return sum(supports) / len(supports)


def _summarize_votes(
    results: list[JudgeResult],
) -> tuple[list[str], str, float, JudgeResult]:
    """Summarize votes and confidence for a set of results.

    Args:
        results: List of JudgeResult entries.

    Returns:
        Tuple of (votes, winner, avg_confidence, winning_result).

    Raises:
        ValueError: If no results are provided.
    """
    if not results:
        raise ValueError("At least one judge result is required")
    votes = [r.winner for r in results]
    winner = "A" if votes.count("A") > votes.count("B") else "B"
    avg_confidence = _aggregate_confidence(results, winner)
    winning_result = next(r for r in results if r.winner == winner)
    return votes, winner, avg_confidence, winning_result


def _confidence_tiebreak(
    primary_result: JudgeResult,
    second_result: JudgeResult,
    primary_judge: str,
    second_judge: str,
) -> tuple[JudgeResult, str, list[str]]:
    """Resolve tiebreak by higher confidence.

    Args:
        primary_result: Primary judge result.
        second_result: Second judge result.
        primary_judge: Primary judge ID.
        second_judge: Second judge ID.

    Returns:
        Tuple of (winner_result, winner_judge, audit_judges).
    """
    if primary_result.confidence >= second_result.confidence:
        return primary_result, primary_judge, [second_judge]
    return second_result, second_judge, [primary_judge]


async def _run_parallel_judges(
    client: LLMClient,
    context: MatchContext,
    judges: list[str],
) -> list[JudgeResult]:
    """Run multiple judges in parallel and return their results.

    Args:
        client: Async LLM client.
        context: MatchContext with essay data and settings.
        judges: Judge model IDs.

    Returns:
        List of JudgeResult entries.
    """
    tasks = [
        judge_match(
            client,
            context,
            judge,
            None,
        )
        for judge in judges
    ]
    return await asyncio.gather(*tasks)


async def judge_match(
    client: LLMClient,
    context: MatchContext,
    judge_model: str,
    rotation: JudgeRotation | None = None,
    _attempted_judges: set[str] | None = None,
) -> JudgeResult:
    """Have a judge evaluate a pair of essays.

    Args:
        client: Async LLM client.
        context: MatchContext with essay data and settings.
        judge_model: Judge model ID.
        rotation: Optional judge rotation for fallback on parse failure.
        _attempted_judges: Internal set of already-tried judges.

    Returns:
        Parsed JudgeResult.
    """
    attempted = _attempted_judges or {judge_model}

    try:
        response = await _request_judgment(client, judge_model, context, False)
        return parse_judge_response(response)
    except ValueError:
        logger.warning("judge_parse_failed", judge=judge_model, retrying=True)

        try:
            response = await _request_judgment(client, judge_model, context, True)
            return _parse_with_repair(response)
        except ValueError:
            fallback = _next_fallback_judge(rotation, attempted)
            if fallback:
                logger.warning(
                    "judge_fallback",
                    failed_judge=judge_model,
                    fallback_judge=fallback,
                )
                attempted.add(fallback)
                return await judge_match(
                    client,
                    context,
                    fallback,
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

    Raises:
        ValueError: If no judges are available or the audit threshold is missing.
    """
    if not rotation.judge_models:
        raise ValueError("At least one judge is required for audit mode")
    context = MatchContext(
        essay_a_id=essay_a_id,
        essay_b_id=essay_b_id,
        essay_a=essay_a,
        essay_b=essay_b,
        max_tokens=max_tokens,
        temperature=temperature,
        audit_threshold=audit_threshold,
    )
    # Primary judgment
    primary_judge = rotation.next_judge()
    primary_result = await judge_match(
        client,
        context,
        primary_judge,
        rotation,
    )

    logger.info(
        "primary_judgment",
        judge=primary_judge,
        winner=primary_result.winner,
        confidence=primary_result.confidence,
    )

    threshold = context.audit_threshold
    if threshold is None:
        raise ValueError("audit_threshold must be set on MatchContext")
    if primary_result.confidence >= threshold:
        return _build_match_result(
            context,
            winner=primary_result.winner,
            confidence=primary_result.confidence,
            reasons=primary_result.reasons,
            winner_edge=primary_result.winner_edge,
            primary_judge=primary_judge,
            final_decision="primary",
        )

    audit_judges = rotation.get_audit_judges(primary_judge)
    if not audit_judges:
        return _build_match_result(
            context,
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
        context,
        second_judge,
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
        avg_confidence = _average_confidence([primary_result, second_result])
        return _build_match_result(
            context,
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
        winner_result, winner_judge, audit = _confidence_tiebreak(
            primary_result, second_result, primary_judge, second_judge
        )
        return _build_match_result(
            context,
            winner=winner_result.winner,
            confidence=winner_result.confidence,
            reasons=winner_result.reasons,
            winner_edge=winner_result.winner_edge,
            primary_judge=winner_judge,
            audit_judges=audit,
            final_decision="confidence_tiebreak",
        )

    third_judge = audit_judges[1]
    third_result = await judge_match(
        client,
        context,
        third_judge,
        rotation,
    )

    logger.info(
        "tiebreak_judgment",
        judge=third_judge,
        winner=third_result.winner,
        confidence=third_result.confidence,
    )

    _, winner, avg_confidence, winning_result = _summarize_votes(
        [primary_result, second_result, third_result]
    )

    return _build_match_result(
        context,
        winner=winner,
        confidence=avg_confidence,
        reasons=winning_result.reasons,
        winner_edge=winning_result.winner_edge,
        primary_judge=primary_judge,
        audit_judges=[second_judge, third_judge],
        final_decision="majority",
    )


async def run_match_parallel_majority(
    client: LLMClient,
    essay_a: str,
    essay_b: str,
    essay_a_id: str,
    essay_b_id: str,
    primary_judges: list[str],
    sub_judges: list[str],
    confidence_threshold: float,
    max_tokens: int,
    temperature: float,
    primary_judge_count: int = PRIMARY_JUDGE_COUNT,
    sub_judge_count: int = SUB_JUDGE_COUNT,
) -> MatchResult:
    """Run match with parallel judges and majority voting.

    Args:
        client: Async LLM client.
        essay_a: Essay A text.
        essay_b: Essay B text.
        essay_a_id: Essay A identifier.
        essay_b_id: Essay B identifier.
        primary_judges: List of primary judge models.
        sub_judges: List of sub-judge models for low-confidence expansion.
        confidence_threshold: Below this, add sub-judges.
        max_tokens: Max tokens for responses.
        temperature: Sampling temperature.
        primary_judge_count: How many primary judges to use (default 3).
        sub_judge_count: How many sub-judges to add on low confidence (default 2).

    Returns:
        MatchResult with majority decision.

    Raises:
        ValueError: If the audit threshold is missing.
    """
    context = MatchContext(
        essay_a_id=essay_a_id,
        essay_b_id=essay_b_id,
        essay_a=essay_a,
        essay_b=essay_b,
        max_tokens=max_tokens,
        temperature=temperature,
        audit_threshold=confidence_threshold,
    )
    judges_to_use = (
        primary_judges[:primary_judge_count]
        if len(primary_judges) >= primary_judge_count
        else primary_judges
    )
    if not judges_to_use:
        raise ValueError("At least one primary judge is required")

    if len(judges_to_use) < primary_judge_count:
        logger.warning(
            "insufficient_primary_judges",
            have=len(judges_to_use),
            need=primary_judge_count,
        )

    # Run judges in parallel
    results = await _run_parallel_judges(
        client,
        context,
        judges_to_use,
    )

    # Majority vote
    votes, winner, avg_confidence, winning_result = _summarize_votes(results)

    logger.info(
        "parallel_majority_vote",
        judges=judges_to_use,
        votes=votes,
        winner=winner,
        avg_confidence=avg_confidence,
    )

    # Check if we need sub-judges
    threshold = context.audit_threshold
    if threshold is None:
        raise ValueError("audit_threshold must be set on MatchContext")
    if avg_confidence < threshold and sub_judges:
        logger.info("expanding_with_sub_judges", threshold=threshold)

        sub_to_use = (
            sub_judges[:sub_judge_count] if len(sub_judges) >= sub_judge_count else sub_judges
        )
        sub_results = await _run_parallel_judges(
            client,
            context,
            sub_to_use,
        )

        # Combine all results (5 total)
        all_results = results + sub_results
        all_votes, winner, avg_confidence, winning_result = _summarize_votes(all_results)

        logger.info(
            "expanded_vote",
            all_judges=judges_to_use + sub_to_use,
            all_votes=all_votes,
            winner=winner,
            avg_confidence=avg_confidence,
        )

        all_judges = judges_to_use + sub_to_use
        return _build_match_result(
            context,
            winner=winner,
            confidence=avg_confidence,
            reasons=winning_result.reasons,
            winner_edge=winning_result.winner_edge,
            primary_judge=all_judges[0],
            audit_judges=all_judges[1:],
            final_decision="parallel_majority_5",
        )

    return _build_match_result(
        context,
        winner=winner,
        confidence=avg_confidence,
        reasons=winning_result.reasons,
        winner_edge=winning_result.winner_edge,
        primary_judge=judges_to_use[0],
        audit_judges=judges_to_use[1:],
        final_decision="parallel_majority_3",
    )
