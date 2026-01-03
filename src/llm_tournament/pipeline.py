"""Pipeline orchestration for LLM Tournament."""

from __future__ import annotations

import asyncio

import structlog

from llm_tournament.core.config import TournamentConfig, calculate_rounds, model_slug
from llm_tournament.ranking import RankingSystem, create_ranking_system
from llm_tournament.services.aggregation import AggregationService
from llm_tournament.services.analysis import AnalysisService
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.match import (
    Candidate,
    JudgeRotation,
    create_candidates_v0,
    create_candidates_v1,
)
from llm_tournament.services.match.service import MatchService
from llm_tournament.services.reporting import (
    build_rating_objects,
    generate_critic_metrics,
    generate_writer_aggregate,
)
from llm_tournament.services.storage import TournamentStore
from llm_tournament.services.submission import SubmissionService

logger = structlog.get_logger()


class TournamentPipeline:
    """Orchestrates the complete tournament pipeline with async execution."""

    def __init__(
        self,
        config: TournamentConfig,
        client: LLMClient,
        store: TournamentStore,
        max_topics: int | None = None,
        max_writers: int | None = None,
        max_critics: int | None = None,
        max_concurrency: int = 5,
    ) -> None:
        """Initialize pipeline.

        Args:
            config: Tournament configuration.
            client: Async LLM client.
            store: Tournament store for persistence.
            max_topics: Optional limit on topics to process.
            max_writers: Optional limit on writers.
            max_critics: Optional limit on critics.
            max_concurrency: Maximum concurrent API calls.
        """
        self.config = config
        self.client = client
        self.store = store
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Apply limits
        self.topics = config.topics[:max_topics] if max_topics else config.topics
        self.writers = config.writers[:max_writers] if max_writers else config.writers
        self.critics = config.critics[:max_critics] if max_critics else config.critics
        self.judges = config.judges

        # Slugify model IDs
        self.writer_slugs = [model_slug(w) for w in self.writers]
        self.critic_slugs = [model_slug(c) for c in self.critics]

        # Initialize services
        self.submission_service = SubmissionService(
            config, client, store, self._semaphore
        )
        self.match_service = MatchService(config, client, store)
        self.analysis_service = AnalysisService(config, client, store, self._semaphore)
        self.aggregation_service = AggregationService(
            config, client, store, self._semaphore
        )

    async def run(self) -> None:
        """Execute complete tournament pipeline."""
        logger.info(
            "pipeline_start", topics=len(self.topics), writers=len(self.writers)
        )

        for topic in self.topics:
            logger.info("processing_topic", title=topic.title)
            await self._process_topic(topic.slug)

        # Stage 6: Cross-Topic Aggregation
        logger.info("stage_aggregation")
        await self.aggregation_service.run_aggregation()

        logger.info("pipeline_complete")

    async def _process_topic(self, topic_slug: str) -> None:
        """Process a single topic through all stages."""
        topic = next(t for t in self.topics if t.slug == topic_slug)

        # Stage 1: Generation
        logger.info("stage_generation", topic=topic_slug)
        await self.submission_service.run_generation_batch(topic, self.writers)

        if not self.config.simple_mode:
            # Stage 2: Critique
            logger.info("stage_critique", topic=topic_slug)
            await self.submission_service.run_critique_batch(
                topic, self.writers, self.critics
            )

            # Stage 3: Revision
            logger.info("stage_revision", topic=topic_slug)
            await self.submission_service.run_revision_batch(
                topic, self.writers, self.critics
            )

        # Stage 4: Ranking
        logger.info("stage_ranking", topic=topic_slug)
        await self._run_ranking(topic_slug)

        # Stage 5: Analysis
        logger.info("stage_analysis", topic=topic_slug)
        await self.analysis_service.run_analysis(topic_slug)

    async def _run_ranking(self, topic_slug: str) -> None:
        """Run pairwise tournament ranking."""
        if self.config.simple_mode:
            candidates = create_candidates_v0(
                self.writer_slugs, self.config.ranking.initial_elo
            )
            version = "v0"
        else:
            candidates = create_candidates_v1(
                self.writer_slugs, self.critic_slugs, self.config.ranking.initial_elo
            )
            version = "v1"

        # Auto-calculate rounds if not specified
        rounds = self.config.ranking.rounds
        if rounds is None:
            rounds = calculate_rounds(len(candidates))
            logger.info(
                "auto_calculated_rounds", candidates=len(candidates), rounds=rounds
            )

        ranking_system = create_ranking_system(self.config)
        ranking_system.initialize([c.id for c in candidates])

        rotation = JudgeRotation(self.judges)
        for round_num in range(1, rounds + 1):
            await self.match_service.run_ranking_round(
                topic_slug, round_num, candidates, ranking_system, rotation, version
            )

        await self._save_leaderboard(topic_slug, candidates, ranking_system)
        await self._save_aggregates(topic_slug, candidates, ranking_system)

    async def _save_leaderboard(
        self,
        topic_slug: str,
        candidates: list[Candidate],
        ranking_system: RankingSystem,
    ) -> None:
        """Generate and save leaderboard."""
        logger.info("generating_leaderboard")

        leaderboard = ranking_system.get_leaderboard()
        rating_objects = build_rating_objects(
            leaderboard, candidates, self.config.ranking.algorithm
        )

        for rating_obj in rating_objects:
            await self.store.db.save_rating(topic_slug, rating_obj)

        await self.store.reports.save_ranking_output(
            topic_slug, rating_objects, ranking_system
        )
        await self.store.reports.export_to_json(topic_slug, rating_objects)

    async def _save_aggregates(
        self,
        topic_slug: str,
        candidates: list[Candidate],
        ranking_system: RankingSystem,
    ) -> None:
        """Generate and save aggregate statistics."""
        writer_report = generate_writer_aggregate(candidates, ranking_system)
        await self.store.reports.save_report(
            topic_slug, "writer_aggregate.md", writer_report
        )

        if not self.config.simple_mode:
            critic_report = generate_critic_metrics(candidates, ranking_system)
            await self.store.reports.save_report(
                topic_slug, "critic_metrics.md", critic_report
            )


async def run_tournament(
    config: TournamentConfig,
    client: LLMClient,
    run_id: str | None = None,
    max_topics: int | None = None,
    max_writers: int | None = None,
    max_critics: int | None = None,
    max_concurrency: int = 5,
) -> TournamentStore:
    """Convenience function to run a complete tournament.

    Args:
        config: Tournament configuration.
        client: Async LLM client.
        run_id: Optional run ID.
        max_topics: Optional topic limit.
        max_writers: Optional writer limit.
        max_critics: Optional critic limit.
        max_concurrency: Maximum concurrent API calls.

    Returns:
        TournamentStore with all artifacts.
    """
    store = TournamentStore(config, run_id)
    pipeline = TournamentPipeline(
        config, client, store, max_topics, max_writers, max_critics, max_concurrency
    )
    await pipeline.run()
    return store
