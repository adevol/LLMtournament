"""Unified tournament storage layer with file, database, and report operations."""

from __future__ import annotations

import asyncio
import csv
import gc
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, col, create_engine, select

from llm_tournament import __version__
from llm_tournament.core.config import TournamentConfig
from llm_tournament.models import Match, Rating
from llm_tournament.services.storage.repository import AsyncRepository

logger = structlog.get_logger()


def _normalize_reasons(reasons: Any) -> list[str]:
    """Normalize reasons field to a list of strings."""
    if isinstance(reasons, str):
        return [reasons]
    if reasons is None:
        return []
    if isinstance(reasons, list):
        return [str(r) for r in reasons]
    return [str(reasons)]


class TournamentStore:
    """Unified persistence layer for tournament data.

    Handles:
    - File-based storage for essays, feedback, revisions (Markdown)
    - SQLModel-based storage for matches, ratings (DuckDB/SQL)
    - Report generation and export (Markdown, CSV, JSON)
    """

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
        self._repository: AsyncRepository[Any] | None = None
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
        self._repository = AsyncRepository(self._engine)
        SQLModel.metadata.create_all(self._engine)

    def _get_repository(self) -> AsyncRepository[Any]:
        if self._repository is None:
            raise RuntimeError("Database repository is not initialized")
        return self._repository

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

    # ==================== Directory helpers ====================

    def topic_dir(self, topic_slug: str) -> Path:
        """Get or create topic directory."""
        path = self.base_dir / topic_slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _topic_subdir(self, topic_slug: str, subdir: str) -> Path:
        """Get or create a topic subdirectory."""
        path = self.topic_dir(topic_slug) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _essay_path(self, topic_slug: str, essay_id: str, version: str) -> Path:
        """Build path to an essay markdown file."""
        subdir = "v0" if version == "v0" else "v1"
        return self._topic_subdir(topic_slug, subdir) / f"{essay_id}.md"

    def _feedback_path(self, topic_slug: str, writer_slug: str, critic_slug: str) -> Path:
        """Build path to a feedback markdown file."""
        return self._topic_subdir(topic_slug, "feedback") / f"{writer_slug}__{critic_slug}.md"

    def _revision_path(self, topic_slug: str, writer_slug: str, critic_slug: str) -> Path:
        """Build path to a revised essay markdown file."""
        return self._topic_subdir(topic_slug, "v1") / f"{writer_slug}__{critic_slug}.md"

    def _ranking_path(self, topic_slug: str, filename: str) -> Path:
        """Build path to a ranking artifact file."""
        return self._topic_subdir(topic_slug, "ranking") / filename

    def _aggregation_path(self, filename: str) -> Path:
        """Build path to a cross-topic aggregation artifact."""
        output_dir = self.base_dir / "final_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        path = output_dir / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    @staticmethod
    def _write_text(path: Path, content: str) -> Path:
        """Write text content to a file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read text content from a file path."""
        return path.read_text(encoding="utf-8")

    async def _write_text_async(self, path: Path, content: str) -> Path:
        """Async wrapper for text file writes."""
        return await asyncio.to_thread(self._write_text, path, content)

    async def _read_text_async(self, path: Path) -> str:
        """Async wrapper for text file reads."""
        return await asyncio.to_thread(self._read_text, path)

    # ==================== Essay operations ====================

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> Path:
        """Save an essay to file."""
        path = self._essay_path(topic_slug, writer_slug, version)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_essay", path=str(saved_path))
        return saved_path

    async def load_essay(self, topic_slug: str, essay_id: str, version: str) -> str:
        """Load an essay from file."""
        path = self._essay_path(topic_slug, essay_id, version)
        return await self._read_text_async(path)

    async def save_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save feedback to file."""
        path = self._feedback_path(topic_slug, writer_slug, critic_slug)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_feedback", path=str(saved_path))
        return saved_path

    async def load_feedback(self, topic_slug: str, writer_slug: str, critic_slug: str) -> str:
        """Load feedback from file."""
        path = self._feedback_path(topic_slug, writer_slug, critic_slug)
        return await self._read_text_async(path)

    async def save_revision(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save revised essay to file."""
        path = self._revision_path(topic_slug, writer_slug, critic_slug)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_revision", path=str(saved_path))
        return saved_path

    # ==================== Match/Rating operations ====================

    async def save_match(self, topic_slug: str, match_data: Any) -> None:
        """Save a match result to database and JSONL backup."""
        data = match_data.model_dump() if hasattr(match_data, "model_dump") else dict(match_data)

        if "topic_slug" not in data:
            data["topic_slug"] = topic_slug

        data["reasons"] = _normalize_reasons(data.get("reasons"))

        def _db_save(session: Session) -> None:
            match = Match.model_validate(data)
            session.add(match)
            session.commit()

        await self._get_repository()._run_session(_db_save)

        def _save_jsonl() -> None:
            jsonl_path = self._ranking_path(topic_slug, "matches.jsonl")
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")

        await asyncio.to_thread(_save_jsonl)

    async def get_matches_for_essay(self, topic_slug: str, essay_id: str) -> list[dict[str, Any]]:
        """Get all matches involving an essay."""

        def _get(session: Session) -> list[dict[str, Any]]:
            statement = select(Match).where(
                Match.topic_slug == topic_slug,
                (Match.essay_a_id == essay_id) | (Match.essay_b_id == essay_id),
            )
            results = session.exec(statement).all()
            return [m.model_dump() for m in results]

        return await self._get_repository()._run_session(_get)

    async def save_rating(self, topic_slug: str, rating_data: Any) -> None:
        """Save or update a rating in the database."""
        data = rating_data.model_dump() if hasattr(rating_data, "model_dump") else dict(rating_data)

        if "topic_slug" not in data:
            data["topic_slug"] = topic_slug

        def _save(session: Session) -> None:
            statement = select(Rating).where(
                Rating.topic_slug == topic_slug,
                Rating.candidate_id == data["candidate_id"],
            )
            existing = session.exec(statement).first()
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                session.add(existing)
            else:
                rating = Rating.model_validate(data)
                session.add(rating)
            session.commit()

        await self._get_repository()._run_session(_save)

    async def get_leaderboard(self, topic_slug: str) -> list[Any]:
        """Get leaderboard sorted by rating."""

        def _get(session: Session) -> list[Any]:
            statement = (
                select(Rating)
                .where(Rating.topic_slug == topic_slug)
                .order_by(col(Rating.rating).desc())
            )
            results = session.exec(statement).all()
            return list(results)

        return await self._get_repository()._run_session(_get)

    async def get_all_ratings(self) -> list[Any]:
        """Get all ratings across all topics."""

        def _get(session: Session) -> list[Any]:
            statement = select(Rating)
            results = session.exec(statement).all()
            return list(results)

        return await self._get_repository()._run_session(_get)

    # ==================== Report operations ====================

    async def save_report(self, topic_slug: str, filename: str, content: str) -> None:
        """Save a generic report file (Markdown/Text)."""
        path = self._ranking_path(topic_slug, filename)
        await self._write_text_async(path, content)
        logger.debug("saved_report", path=str(path))

    async def save_ranking_output(
        self, topic_slug: str, leaderboard: list[Any], _ranking_system: Any
    ) -> None:
        """Save leaderboard to text/csv files for human consumption."""

        def _save() -> None:
            md_path = self._ranking_path(topic_slug, "leaderboard.md")
            with md_path.open("w", encoding="utf-8") as f:
                f.write(f"# Leaderboard: {topic_slug}\n\n")
                f.write("| Rank | Candidate | Rating | Matches | Wins | Losses |\n")
                f.write("|---|---|---|---|---|---|\n")
                for i, r in enumerate(leaderboard, 1):
                    f.write(
                        f"| {i} | {r.candidate_id} | {r.rating:.1f} | "
                        f"{r.matches} | {r.wins} | {r.losses} |\n"
                    )

            csv_path = self._ranking_path(topic_slug, "leaderboard.csv")
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rank",
                        "candidate_id",
                        "rating",
                        "matches",
                        "wins",
                        "losses",
                        "mu",
                        "sigma",
                    ]
                )
                for i, r in enumerate(leaderboard, 1):
                    mu = getattr(r, "mu", "")
                    sigma = getattr(r, "sigma", "")
                    writer.writerow(
                        [
                            i,
                            r.candidate_id,
                            f"{r.rating:.2f}",
                            r.matches,
                            r.wins,
                            r.losses,
                            mu,
                            sigma,
                        ]
                    )

        await asyncio.to_thread(_save)

    async def export_to_json(self, topic_slug: str, leaderboard: list[Any]) -> None:
        """Export leaderboard to JSON for dashboard consumption."""

        def _save() -> None:
            json_path = self._ranking_path(topic_slug, "leaderboard.json")
            data = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in leaderboard]
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        await asyncio.to_thread(_save)

    async def save_aggregation_report(self, filename: str, content: str) -> None:
        """Save a cross-topic aggregation report file."""
        path = self._aggregation_path(filename)
        await self._write_text_async(path, content)
        logger.debug("saved_aggregation_report", path=str(path))

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
