"""Analysis service for LLM Tournament.

Handles both per-topic analysis of top candidates and cross-topic aggregation.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import structlog

from llm_tournament.core.config import TournamentConfig
from llm_tournament.core.slug import SlugGenerator
from llm_tournament.models import Rating
from llm_tournament.prompts import analysis_system_prompt, analysis_user_prompt
from llm_tournament.prompts.aggregation import (
    cross_topic_insights_system_prompt,
    cross_topic_insights_user_prompt,
    model_profile_system_prompt,
    model_profile_user_prompt,
)
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
    opponent = match["essay_b_id"] if match["essay_a_id"] == essay_id else match["essay_a_id"]
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

    edge_label = "Edge" if won else "Weakness"
    return (
        f"vs {opponent[:20]}: {'WON' if won else 'LOST'}\n"
        f"{edge_label}: {match['winner_edge']}\n"
        f"Reasons: {', '.join(reasons[:2])}"
    )


class AnalysisService:
    """Handles post-ranking analysis and cross-topic aggregation."""

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
        self._slugger = SlugGenerator(max_length=None)

    # ==================== Per-topic analysis ====================

    async def run_analysis(self, topic_slug: str) -> None:
        """Run post-ranking analysis for top candidates.

        Args:
            topic_slug: Topic slug.
        """
        leaderboard = await self.store.get_leaderboard(topic_slug)

        if not leaderboard:
            logger.warning("no_leaderboard_for_analysis", topic=topic_slug)
            return

        top_candidates = leaderboard[: self.config.analysis_top_k]
        await asyncio.gather(*[self._analyze_candidate(topic_slug, r) for r in top_candidates])

    async def _analyze_candidate(self, topic_slug: str, rating: Rating) -> None:
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
                self.config.judge_tokens,
                self.config.judge_temp,
            )
            safe_essay_id = self._slugger.safe_id(essay_id)
            await self.store.save_report(
                topic_slug, f"analysis_{safe_essay_id}.md", analysis.content
            )

    # ==================== Cross-topic aggregation ====================

    async def run_aggregation(self) -> None:
        """Run complete set of cross-topic aggregation tasks."""
        ratings = await self.store.get_all_ratings()
        if not ratings:
            logger.warning("no_ratings_for_aggregation")
            return

        ranking_summary, model_results = self._compute_aggregates(ratings)
        await self.store.save_aggregation_report("aggregate_ranking.md", ranking_summary)

        await self._generate_model_profiles(model_results)
        await self._generate_cross_topic_insights(ranking_summary)

    def _compute_aggregates(self, ratings: list[Rating]) -> tuple[str, dict]:
        """Compute aggregate stats and organize data structure."""
        model_results = defaultdict(list)
        model_avg_scores = defaultdict(list)

        topic_groups = defaultdict(list)
        for r in ratings:
            topic_groups[r.topic_slug].append(r)

        for topic, group in topic_groups.items():
            group.sort(key=lambda x: x.rating, reverse=True)
            for rank, r in enumerate(group, 1):
                model_id = r.candidate_id
                model_results[model_id].append(
                    {
                        "topic": topic,
                        "rank": f"{rank}/{len(group)}",
                        "score": r.rating,
                        "summary": f"Rating: {r.rating:.1f}, Rank: {rank}",
                    }
                )
                model_avg_scores[model_id].append(r.rating)

        aggregates = []
        for model_id, scores in model_avg_scores.items():
            avg_score = sum(scores) / len(scores)
            aggregates.append((model_id, avg_score, len(scores)))

        aggregates.sort(key=lambda x: x[1], reverse=True)

        lines = [
            "# Aggregate Ranking\n",
            "| Rank | Model | Avg Rating | Topics |",
            "|---|---|---|---|",
        ]
        for i, (model, score, count) in enumerate(aggregates, 1):
            lines.append(f"| {i} | {model} | {score:.2f} | {count} |")

        return "\n".join(lines), model_results

    async def _generate_model_profiles(self, model_results: dict) -> None:
        """Generate profile for each model."""
        tasks = []
        for model_id, results in model_results.items():
            tasks.append(self._profile_single_model(model_id, results))

        await asyncio.gather(*tasks)

    async def _profile_single_model(self, model_id: str, results: list[dict]) -> None:
        """Generate profile for one model."""
        async with self._semaphore:
            messages = [
                {"role": "system", "content": model_profile_system_prompt()},
                {
                    "role": "user",
                    "content": model_profile_user_prompt(model_id, results),
                },
            ]
            response = await self.client.complete(
                self.config.judges[0],
                messages,
                self.config.judge_tokens,
                self.config.judge_temp,
            )
            safe_model_id = self._slugger.safe_id(model_id)
            filename = f"model_profiles/{safe_model_id}.json"
            await self.store.save_aggregation_report(filename, response.content)

    async def _generate_cross_topic_insights(self, ranking_summary: str) -> None:
        """Generate high-level tournament insights."""
        model_profiles_text = "(Model profiles generated in individual files)"

        async with self._semaphore:
            messages = [
                {"role": "system", "content": cross_topic_insights_system_prompt()},
                {
                    "role": "user",
                    "content": cross_topic_insights_user_prompt(
                        ranking_summary, model_profiles_text
                    ),
                },
            ]

            response = await self.client.complete(
                self.config.judges[0],
                messages,
                self.config.judge_tokens,
                self.config.judge_temp,
            )

            await self.store.save_aggregation_report("cross_topic_insights.json", response.content)
