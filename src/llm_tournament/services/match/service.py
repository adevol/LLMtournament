"""Match service for running tournament rounds."""

from __future__ import annotations

import structlog

from llm_tournament.core.config import TournamentConfig
from llm_tournament.models import Match
from llm_tournament.ranking import RankingSystem
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.match import (
    Candidate,
    JudgeRotation,
    MatchResult,
    run_match_parallel_majority,
    run_match_with_audit,
    swiss_pairing,
)
from llm_tournament.services.storage import TournamentStore

logger = structlog.get_logger()


class MatchService:
    """Orchestrates pairing generations, match execution, and result persistence.

    Coordinates between the pairing algorithm, LLM judges, ranking system, and
    storage layer to run complete tournament rounds.
    """

    def __init__(
        self,
        config: TournamentConfig,
        client: LLMClient,
        store: TournamentStore,
    ) -> None:
        """Initialize match service.

        Args:
            config: Tournament configuration.
            client: LLM client for judge calls.
            store: Storage layer for essays and matches.
        """
        self.config = config
        self.client = client
        self.store = store

    async def run_ranking_round(
        self,
        topic_slug: str,
        round_num: int,
        candidates: list[Candidate],
        ranking_system: RankingSystem,
        rotation: JudgeRotation,
        version: str,
    ) -> None:
        """Run a single round of Swiss pairing and judging.

        Updates candidate ratings from the ranking system, generates pairings,
        runs matches with audit judges, and persists results.

        Args:
            topic_slug: Topic identifier for storage paths.
            round_num: Current round number (used for seed offset).
            candidates: List of candidates to pair and judge.
            ranking_system: Rating system to query and update.
            rotation: Judge rotation strategy.
            version: Essay version to load ("v0" or "v1").
        """
        logger.info("ranking_round", round=round_num)

        self._sync_ratings(candidates, ranking_system)
        pairs, bye_recipient = swiss_pairing(
            candidates, seed=self.config.seed + round_num
        )
        logger.info(
            "round_pairs",
            count=len(pairs),
            bye=bye_recipient.id if bye_recipient else None,
        )

        for candidate_a, candidate_b in pairs:
            result = await self._run_single_match(
                topic_slug, candidate_a, candidate_b, rotation, version
            )
            self._apply_result(candidate_a, candidate_b, result, ranking_system)
            await self._persist_match(topic_slug, result)

    def _sync_ratings(
        self, candidates: list[Candidate], ranking_system: RankingSystem
    ) -> None:
        """Update candidate ratings from ranking system before pairing."""
        for c in candidates:
            c.rating = ranking_system.get_rating(c.id)

    async def _run_single_match(
        self,
        topic_slug: str,
        candidate_a: Candidate,
        candidate_b: Candidate,
        rotation: JudgeRotation,
        version: str,
    ) -> MatchResult:
        """Load essays and run judged match using configured method."""
        essay_a = await self.store.load_essay(topic_slug, candidate_a.id, version)
        essay_b = await self.store.load_essay(topic_slug, candidate_b.id, version)

        if self.config.ranking.judging_method == "parallel_majority":
            primary = self.config.ranking.primary_judges or self.config.judges
            count = self.config.ranking.primary_judge_count
            subs = self.config.ranking.sub_judges or [
                j for j in self.config.judges if j not in primary[:count]
            ]
            return await run_match_parallel_majority(
                self.client,
                essay_a,
                essay_b,
                candidate_a.id,
                candidate_b.id,
                primary,
                subs,
                self.config.ranking.audit_confidence_threshold,
                self.config.token_caps.judge_tokens,
                self.config.temperatures.judge,
                self.config.ranking.primary_judge_count,
                self.config.ranking.sub_judge_count,
            )

        return await run_match_with_audit(
            self.client,
            essay_a,
            essay_b,
            candidate_a.id,
            candidate_b.id,
            rotation,
            self.config.ranking.audit_confidence_threshold,
            self.config.token_caps.judge_tokens,
            self.config.temperatures.judge,
        )

    def _apply_result(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        result: MatchResult,
        ranking_system: RankingSystem,
    ) -> None:
        """Update rankings and opponent tracking from match result."""
        winner, loser = (
            (candidate_a, candidate_b)
            if result.winner == "A"
            else (candidate_b, candidate_a)
        )
        ranking_system.update(winner.id, loser.id, result.confidence)

        candidate_a.played_against.add(candidate_b.id)
        candidate_b.played_against.add(candidate_a.id)

        logger.debug(
            "match_complete",
            a=candidate_a.id[:20],
            b=candidate_b.id[:20],
            winner=result.winner,
        )

    async def _persist_match(self, topic_slug: str, result: MatchResult) -> None:
        """Save match result to storage."""
        match_record = Match(
            essay_a_id=result.essay_a_id,
            essay_b_id=result.essay_b_id,
            winner=result.winner,
            confidence=result.confidence,
            reasons=result.reasons,
            winner_edge=result.winner_edge,
            primary_judge=result.primary_judge,
            audit_judges=result.audit_judges,
            final_decision=result.final_decision,
            timestamp=result.timestamp,
        )
        await self.store.save_match(topic_slug, match_record)
