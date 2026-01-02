"""Post-ranking analysis service for LLM Tournament."""

from __future__ import annotations

import asyncio
import json

import structlog

from llm_tournament.core.config import TournamentConfig
from llm_tournament.models import Rating
from llm_tournament.prompts import analysis_system_prompt, analysis_user_prompt
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.storage import TournamentStore

logger = structlog.get_logger()


def build_match_summary(match: dict, essay_id: str) -> str:
    """Build a summary string for a single match.

    Args:
        match: Match data dictionary.
        essay_id: ID of the essay being analyzed.

    Returns:
        Formatted summary string.
    """
    opponent = (
        match["essay_b_id"] if match["essay_a_id"] == essay_id else match["essay_a_id"]
    )
    won = (match["winner"] == "A" and match["essay_a_id"] == essay_id) or (
        match["winner"] == "B" and match["essay_b_id"] == essay_id
    )

    # Parse reasons (may be string or list)
    reasons = match["reasons"]
    if isinstance(reasons, str):
        if reasons.strip().startswith("["):
            try:
                reasons = json.loads(reasons)
            except (json.JSONDecodeError, TypeError):
                reasons = [reasons]
        else:
            reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    return (
        f"vs {opponent[:20]}: {'WON' if won else 'LOST'}\n"
        f"Edge: {match['winner_edge']}\n"
        f"Reasons: {', '.join(reasons[:2])}"
    )


class AnalysisService:
    """Handles post-ranking analysis of top candidates."""

    def __init__(
        self,
        config: TournamentConfig,
        client: LLMClient,
        store: TournamentStore,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize analysis service.

        Args:
            config: Tournament configuration.
            client: Async LLM client.
            store: Tournament store for persistence.
            semaphore: Concurrency limiter.
        """
        self.config = config
        self.client = client
        self.store = store
        self._semaphore = semaphore

    async def analyze_candidate(self, topic_slug: str, rating: Rating) -> None:
        """Analyze a single candidate's performance.

        Args:
            topic_slug: Topic slug.
            rating: Rating object for the candidate.
        """
        async with self._semaphore:
            essay_id = rating.candidate_id
            matches = await self.store.get_matches_for_essay(topic_slug, essay_id)

            if not matches:
                return

            summaries = [build_match_summary(m, essay_id) for m in matches]

            messages = [
                {"role": "system", "content": analysis_system_prompt()},
                {"role": "user", "content": analysis_user_prompt(essay_id, summaries)},
            ]

            analysis = await self.client.complete(
                self.config.judges[0],
                messages,
                self.config.token_caps.judge_tokens,
                self.config.temperatures.judge,
            )

            safe_id = essay_id.replace("/", "__").replace(":", "_")
            await self.store.save_report(topic_slug, f"analysis_{safe_id}.md", analysis)

    async def run_analysis(self, topic_slug: str) -> None:
        """Run post-ranking analysis for top candidates.

        Args:
            topic_slug: Topic slug.
        """
        leaderboard = await self.store.get_leaderboard(topic_slug)

        if not leaderboard:
            logger.warning("no_leaderboard_for_analysis", topic=topic_slug)
            return

        top_candidates = leaderboard[: self.config.analysis.top_k]
        await asyncio.gather(
            *[self.analyze_candidate(topic_slug, r) for r in top_candidates]
        )
