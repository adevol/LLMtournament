"""Facade storage layer that coordinates focused storage services."""

from __future__ import annotations

import gc
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, create_engine

from llm_tournament import __version__
from llm_tournament.core.config import TournamentConfig

from .essay_store import EssayStore
from .match_repository import MatchRepository
from .paths import StoragePaths
from .rating_repository import RatingRepository
from .report_generator import ReportGenerator

logger = structlog.get_logger()


class TournamentStore:
    """Unified storage facade for tournament data."""

    def __init__(
        self,
        config: TournamentConfig,
        run_id: str | None = None,
    ) -> None:
        """Initialize tournament store.

        Args:
            config: Tournament configuration.
            run_id: Optional run identifier (defaults to timestamp).
        """
        self.config = config
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(config.output_dir) / self.run_id
        self._db_path = self.base_dir / "tournament.duckdb"
        self._engine = None

        self.paths = StoragePaths(self.base_dir)
        self.essays = EssayStore(self.paths)
        self.reports = ReportGenerator(self.paths)
        self.matches: MatchRepository
        self.ratings: RatingRepository

        self._init_directories()
        self._init_db()
        self._save_metadata()

    def _init_directories(self) -> None:
        """Create base output directory."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("store_init", run_id=self.run_id, path=str(self.base_dir))

    def _init_db(self) -> None:
        """Initialize DuckDB database and create tables."""
        db_url = f"duckdb:///{self._db_path}"
        # Use NullPool to avoid connection pooling issues on Windows
        self._engine = create_engine(db_url, poolclass=NullPool)
        SQLModel.metadata.create_all(self._engine)
        self.matches = MatchRepository(self._engine, self.paths)
        self.ratings = RatingRepository(self._engine)

    def _save_metadata(self) -> None:
        """Save config snapshot and run metadata."""
        config_path = self.base_dir / "config_snapshot.yaml"
        with config_path.open("w") as f:
            config_dict = self.config.model_dump(exclude={"api_key", "retriever"})
            yaml.dump(config_dict, f, default_flow_style=False)

        metadata = {
            "run_id": self.run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "seed": self.config.seed,
            "llm_tournament_version": __version__,
            "ranking_algorithm": self.config.ranking.algorithm,
        }
        metadata_path = self.base_dir / "run_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

    # ==================== Backward-compat facade ====================

    def topic_dir(self, topic_slug: str) -> Path:
        return self.paths.topic_dir(topic_slug)

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> Path:
        return await self.essays.save_essay(topic_slug, writer_slug, content, version)

    async def load_essay(self, topic_slug: str, essay_id: str, version: str) -> str:
        return await self.essays.load_essay(topic_slug, essay_id, version)

    async def save_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        return await self.essays.save_feedback(topic_slug, writer_slug, critic_slug, content)

    async def load_feedback(self, topic_slug: str, writer_slug: str, critic_slug: str) -> str:
        return await self.essays.load_feedback(topic_slug, writer_slug, critic_slug)

    async def save_revision(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        return await self.essays.save_revision(topic_slug, writer_slug, critic_slug, content)

    async def save_match(self, topic_slug: str, match_data: Any) -> None:
        await self.matches.save_match(topic_slug, match_data)

    async def get_matches_for_essay(self, topic_slug: str, essay_id: str) -> list[dict[str, Any]]:
        return await self.matches.get_matches_for_essay(topic_slug, essay_id)

    async def save_rating(self, topic_slug: str, rating_data: Any) -> None:
        await self.ratings.save_rating(topic_slug, rating_data)

    async def get_leaderboard(self, topic_slug: str) -> list[Any]:
        return await self.ratings.get_leaderboard(topic_slug)

    async def get_all_ratings(self) -> list[Any]:
        return await self.ratings.get_all_ratings()

    async def save_report(self, topic_slug: str, filename: str, content: str) -> None:
        await self.reports.save_topic_report(topic_slug, filename, content)

    async def save_ranking_output(
        self, topic_slug: str, leaderboard: list[Any], ranking_system: Any
    ) -> None:
        await self.reports.save_ranking_output(topic_slug, leaderboard, ranking_system)

    async def export_to_json(self, topic_slug: str, leaderboard: list[Any]) -> None:
        await self.reports.export_to_json(topic_slug, leaderboard)

    async def save_aggregation_report(self, filename: str, content: str) -> None:
        await self.reports.save_aggregation_report(filename, content)

    # ==================== Lifecycle ====================

    async def close(self) -> None:
        """Dispose of the database engine."""
        if self._engine:
            self._engine.dispose()

    def close_sync(self) -> None:
        """Synchronously dispose of the database engine."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

        gc.collect()
