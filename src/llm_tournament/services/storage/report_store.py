"""Report generation and export utilities."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from llm_tournament.services.storage.file_store import FileStore

logger = structlog.get_logger()


class ReportStore:
    """Report generation and export storage."""

    def __init__(self, base_dir: Path, file_store: FileStore) -> None:
        """Initialize report store.

        Args:
            base_dir: Base directory for aggregation reports.
            file_store: FileStore instance for accessing ranking directories.
        """
        self.base_dir = base_dir
        self._file_store = file_store

    async def save_report(self, topic_slug: str, filename: str, content: str) -> None:
        """Save a generic report file (Markdown/Text)."""

        def _save() -> None:
            ranking_dir = self._file_store.ranking_dir(topic_slug)
            path = ranking_dir / filename
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_report", path=str(path))

        await asyncio.to_thread(_save)

    async def save_ranking_output(
        self, topic_slug: str, leaderboard: list[Any], _ranking_system: Any
    ) -> None:
        """Save leaderboard to text/csv files for human consumption."""

        def _save() -> None:
            ranking_dir = self._file_store.ranking_dir(topic_slug)

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
            ranking_dir = self._file_store.ranking_dir(topic_slug)
            json_path = ranking_dir / "leaderboard.json"
            data = [
                r.model_dump() if hasattr(r, "model_dump") else dict(r)
                for r in leaderboard
            ]
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
