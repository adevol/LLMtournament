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

    def _save_metadata(self) -> None:
        """Save config snapshot and run metadata."""
        config_path = self.base_dir / "config_snapshot.yaml"
        with config_path.open("w") as f:
            config_dict = self.config.model_dump(exclude={"api_key"})
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

    def _v0_dir(self, topic_slug: str) -> Path:
        """Get or create v0 essays directory."""
        path = self.topic_dir(topic_slug) / "v0"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _feedback_dir(self, topic_slug: str) -> Path:
        """Get or create feedback directory."""
        path = self.topic_dir(topic_slug) / "feedback"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _v1_dir(self, topic_slug: str) -> Path:
        """Get or create v1 essays directory."""
        path = self.topic_dir(topic_slug) / "v1"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ranking_dir(self, topic_slug: str) -> Path:
        """Get or create ranking directory."""
        path = self.topic_dir(topic_slug) / "ranking"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==================== Essay operations ====================

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> Path:
        """Save an essay to file."""

        def _save() -> Path:
            directory = self._v0_dir(topic_slug) if version == "v0" else self._v1_dir(topic_slug)
            path = directory / f"{writer_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_essay", path=str(path))
            return path

        return await asyncio.to_thread(_save)

    async def load_essay(self, topic_slug: str, essay_id: str, version: str) -> str:
        """Load an essay from file."""

        def _load() -> str:
            directory = self._v0_dir(topic_slug) if version == "v0" else self._v1_dir(topic_slug)
            path = directory / f"{essay_id}.md"
            return path.read_text(encoding="utf-8")

        return await asyncio.to_thread(_load)

    async def save_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save feedback to file."""

        def _save() -> Path:
            path = self._feedback_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_feedback", path=str(path))
            return path

        return await asyncio.to_thread(_save)

    async def load_feedback(self, topic_slug: str, writer_slug: str, critic_slug: str) -> str:
        """Load feedback from file."""

        def _load() -> str:
            path = self._feedback_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            return path.read_text(encoding="utf-8")

        return await asyncio.to_thread(_load)

    async def save_revision(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save revised essay to file."""

        def _save() -> Path:
            path = self._v1_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_revision", path=str(path))
            return path

        return await asyncio.to_thread(_save)

    # ==================== Match/Rating operations ====================

    async def save_match(self, topic_slug: str, match_data: Any) -> None:
        """Save a match result to database and JSONL backup."""
        data = match_data.model_dump() if hasattr(match_data, "model_dump") else dict(match_data)

        if "topic_slug" not in data:
            data["topic_slug"] = topic_slug

        data["reasons"] = _normalize_reasons(data.get("reasons"))

        def _save() -> None:
            with Session(self._engine) as session:
                match = Match.model_validate(data)
                session.add(match)
                session.commit()

            ranking_dir = self._ranking_dir(topic_slug)
            jsonl_path = ranking_dir / "matches.jsonl"
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")

        await asyncio.to_thread(_save)

    async def get_matches_for_essay(self, topic_slug: str, essay_id: str) -> list[dict[str, Any]]:
        """Get all matches involving an essay."""

        def _get() -> list[dict[str, Any]]:
            with Session(self._engine) as session:
                statement = select(Match).where(
                    Match.topic_slug == topic_slug,
                    (Match.essay_a_id == essay_id) | (Match.essay_b_id == essay_id),
                )
                results = session.exec(statement).all()
                return [m.model_dump() for m in results]

        return await asyncio.to_thread(_get)

    async def save_rating(self, topic_slug: str, rating_data: Any) -> None:
        """Save or update a rating in the database."""
        data = rating_data.model_dump() if hasattr(rating_data, "model_dump") else dict(rating_data)

        if "topic_slug" not in data:
            data["topic_slug"] = topic_slug

        def _save() -> None:
            with Session(self._engine) as session:
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

        await asyncio.to_thread(_save)

    async def get_leaderboard(self, topic_slug: str) -> list[Any]:
        """Get leaderboard sorted by rating."""

        def _get() -> list[Any]:
            with Session(self._engine) as session:
                statement = (
                    select(Rating)
                    .where(Rating.topic_slug == topic_slug)
                    .order_by(col(Rating.rating).desc())
                )
                results = session.exec(statement).all()
                return list(results)

        return await asyncio.to_thread(_get)

    async def get_all_ratings(self) -> list[Any]:
        """Get all ratings across all topics."""

        def _get() -> list[Any]:
            with Session(self._engine) as session:
                statement = select(Rating)
                results = session.exec(statement).all()
                return list(results)

        return await asyncio.to_thread(_get)

    # ==================== Report operations ====================

    async def save_report(self, topic_slug: str, filename: str, content: str) -> None:
        """Save a generic report file (Markdown/Text)."""

        def _save() -> None:
            ranking_dir = self._ranking_dir(topic_slug)
            path = ranking_dir / filename
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_report", path=str(path))

        await asyncio.to_thread(_save)

    async def save_ranking_output(
        self, topic_slug: str, leaderboard: list[Any], _ranking_system: Any
    ) -> None:
        """Save leaderboard to text/csv files for human consumption."""

        def _save() -> None:
            ranking_dir = self._ranking_dir(topic_slug)

            md_path = ranking_dir / "leaderboard.md"
            with md_path.open("w", encoding="utf-8") as f:
                f.write(f"# Leaderboard: {topic_slug}\n\n")
                f.write("| Rank | Candidate | Rating | Matches | Wins | Losses |\n")
                f.write("|---|---|---|---|---|---|\n")
                for i, r in enumerate(leaderboard, 1):
                    f.write(
                        f"| {i} | {r.candidate_id} | {r.rating:.1f} | "
                        f"{r.matches} | {r.wins} | {r.losses} |\n"
                    )

            csv_path = ranking_dir / "leaderboard.csv"
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
            ranking_dir = self._ranking_dir(topic_slug)
            json_path = ranking_dir / "leaderboard.json"
            data = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in leaderboard]
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        await asyncio.to_thread(_save)

    async def save_aggregation_report(self, filename: str, content: str) -> None:
        """Save a cross-topic aggregation report file."""

        def _save() -> None:
            output_dir = self.base_dir / "final_analysis"
            output_dir.mkdir(exist_ok=True, parents=True)
            path = output_dir / filename
            output_dir_subdir = path.parent
            output_dir_subdir.mkdir(exist_ok=True, parents=True)

            path.write_text(content, encoding="utf-8")
            logger.debug("saved_aggregation_report", path=str(path))

        await asyncio.to_thread(_save)

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
