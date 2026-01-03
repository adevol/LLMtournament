"""Aggregation service for cross-topic analysis."""

from __future__ import annotations

import asyncio
from collections import defaultdict

import structlog

from llm_tournament.core.config import TournamentConfig, safe_id
from llm_tournament.models import Rating
from llm_tournament.prompts.aggregation import (
    cross_topic_insights_system_prompt,
    cross_topic_insights_user_prompt,
    model_profile_system_prompt,
    model_profile_user_prompt,
)
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.storage import TournamentStore

logger = structlog.get_logger()


class AggregationService:
    """Handles cross-topic aggregation and analysis."""

    def __init__(
        self,
        config: TournamentConfig,
        client: LLMClient,
        store: TournamentStore,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize aggregation service."""
        self.config = config
        self.client = client
        self.store = store
        self._semaphore = semaphore

    async def run_aggregation(self) -> None:
        """Run complete set of aggregation tasks."""
        ratings = await self.store.db.get_all_ratings()
        if not ratings:
            logger.warning("no_ratings_for_aggregation")
            return

        ranking_summary, model_results = self._compute_aggregates(ratings)
        await self.store.reports.save_aggregation_report(
            "aggregate_ranking.md", ranking_summary
        )

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
                self.config.token_caps.judge_tokens,
                self.config.temperatures.judge,
            )
            safe_model_id = safe_id(model_id)
            filename = f"model_profiles/{safe_model_id}.json"
            await self.store.reports.save_aggregation_report(filename, response)

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
                self.config.token_caps.judge_tokens,
                self.config.temperatures.judge,
            )

            await self.store.reports.save_aggregation_report(
                "cross_topic_insights.json", response
            )
