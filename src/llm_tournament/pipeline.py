"""Pipeline orchestration for LLM Tournament."""

from __future__ import annotations

import asyncio

import structlog

from llm_tournament.core.config import TournamentConfig, calculate_nr_rounds
from llm_tournament.core.progress import TournamentProgress
from llm_tournament.ranking import RankingSystem, create_ranking_system
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
    generate_aggregate_report,
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
        max_concurrency: int = 5,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config: Tournament configuration.
            client: Async LLM client.
            store: Tournament store for persistence.
            max_concurrency: Maximum concurrent API calls.
        """
        self.config = config
        self.client = client
        self.store = store
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.progress = TournamentProgress()

        self.judges = config.judges

        # Slugify model IDs (supports both string and WriterConfig)
        self.writer_slugs = [self.config.get_writer_slug(w) for w in self.config.writers]
        self.critic_slugs = [self.config.get_slug_model(c) for c in self.config.critics]

        # Initialize services
        self.submission_service = SubmissionService(config, client, store, self._semaphore)
        self.match_service = MatchService(config, client, store)
        self.analysis_service = AnalysisService(config, client, store, self._semaphore)

    async def run(self) -> None:
        """Execute complete tournament pipeline."""
        logger.info(
            "pipeline_start",
            topics=len(self.config.topics),
            writers=len(self.config.writers),
        )

        # Process topics with progress tracking
        await self.progress.track_generation(
            self.config.topics,
            lambda topic: self._process_topic(topic.slug),
            description="Processing topics",
        )

        # Cross-Topic Aggregation
        logger.info("stage_aggregation")
        await self.analysis_service.run_aggregation()

        logger.info("pipeline_complete")

    async def _process_topic(self, topic_slug: str) -> None:
        """Process a single topic through all stages."""
        topic = next(t for t in self.config.topics if t.slug == topic_slug)
        logger.info("processing_topic", title=topic.title)

        # Generation
        logger.info("stage_generation", topic=topic_slug)
        await self.submission_service.run_generation_batch(topic, self.config.writers)

        if not self.config.simple_mode:
            # Critique
            logger.info("stage_critique", topic=topic_slug)
            await self.submission_service.run_critique_batch(
                topic, self.config.writers, self.config.critics
            )

            # Revision
            logger.info("stage_revision", topic=topic_slug)
            await self.submission_service.run_revision_batch(
                topic, self.config.writers, self.config.critics
            )

        # Ranking
        logger.info("stage_ranking", topic=topic_slug)
        await self._run_ranking(topic_slug)

        # Analysis
        logger.info("stage_analysis", topic=topic_slug)
        await self.analysis_service.run_analysis(topic_slug)

    def _create_candidates(self) -> tuple[list[Candidate], str]:
        """Create candidates based on simple_mode config."""
        if self.config.simple_mode:
            candidates = create_candidates_v0(self.writer_slugs, self.config.ranking.initial_elo)
            return candidates, "v0"
        candidates = create_candidates_v1(
            self.writer_slugs, self.critic_slugs, self.config.ranking.initial_elo
        )
        return candidates, "v1"

    async def _run_ranking(self, topic_slug: str) -> None:
        """Run pairwise tournament ranking."""
        candidates, version = self._create_candidates()

        # Auto-calculate rounds if not specified
        rounds = self.config.ranking.rounds or calculate_nr_rounds(len(candidates))
        if self.config.ranking.rounds is None:
            logger.info("auto_calculated_rounds", candidates=len(candidates), rounds=rounds)

        ranking_system = create_ranking_system(self.config)
        ranking_system.initialize([c.id for c in candidates])

        rotation = JudgeRotation(self.judges)
        # Track ranking rounds with progress
        await self.progress.track_rounds(
            rounds,
            lambda r: self.match_service.run_ranking_round(
                topic_slug, r, candidates, ranking_system, rotation, version
            ),
            description="Running ranking rounds",
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
            await self.store.save_rating(topic_slug, rating_obj)

        await self.store.save_ranking_output(topic_slug, rating_objects, ranking_system)
        await self.store.export_to_json(topic_slug, rating_objects)

    async def _save_aggregates(
        self,
        topic_slug: str,
        candidates: list[Candidate],
        ranking_system: RankingSystem,
    ) -> None:
        """Generate and save aggregate statistics."""
        writer_report = generate_aggregate_report(
            candidates,
            ranking_system,
            group_by="writer",
            title="Writer Aggregate Rankings",
            headers=("Writer", "Mean Rating", "Variants"),
            max_slug_length=self.config.slug_max_length,
        )
        await self.store.save_report(topic_slug, "writer_aggregate.md", writer_report)

        if not self.config.simple_mode:
            critic_report = generate_aggregate_report(
                candidates,
                ranking_system,
                group_by="critic",
                title="Critic Metrics",
                headers=("Critic", "Mean Rating", "Essays"),
                max_slug_length=self.config.slug_max_length,
                description="Mean rating of essays revised using each critic's feedback.",
            )
            await self.store.save_report(topic_slug, "critic_metrics.md", critic_report)


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
        max_topics: Optional limit on number of topics to run.
        max_writers: Optional limit on number of writers to run.
        max_critics: Optional limit on number of critics to run.
        max_concurrency: Maximum concurrent API calls.

    Returns:
        TournamentStore with all artifacts.
    """
    limits = {
        "max_topics": max_topics,
        "max_writers": max_writers,
        "max_critics": max_critics,
    }
    for limit_name, limit_value in limits.items():
        if limit_value is not None and limit_value <= 0:
            msg = f"{limit_name} must be greater than 0"
            raise ValueError(msg)

    effective_config = config.model_copy(
        update={
            "topics": config.topics[:max_topics] if max_topics is not None else config.topics,
            "writers": config.writers[:max_writers] if max_writers is not None else config.writers,
            "critics": config.critics[:max_critics] if max_critics is not None else config.critics,
        }
    )

    store = TournamentStore(effective_config, run_id)
    pipeline = TournamentPipeline(effective_config, client, store, max_concurrency)
    await pipeline.run()

    if client.total_cost > 0:
        logger.info("cost_summary", total_usd=f"${client.total_cost:.4f}")

    return store
