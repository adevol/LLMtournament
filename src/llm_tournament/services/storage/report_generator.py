"""Output generation for leaderboard and JSON artifacts."""

from __future__ import annotations

import asyncio
import csv
import json
from typing import Any

from .paths import StoragePaths


class ReportGenerator:
    """Generate and persist leaderboard output files."""

    def __init__(self, paths: StoragePaths) -> None:
        self._paths = paths

    @staticmethod
    def _write_text(path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def _write_text_async(self, path, content: str) -> None:
        await asyncio.to_thread(self._write_text, path, content)

    async def save_topic_report(self, topic_slug: str, filename: str, content: str) -> None:
        """Save a topic report file."""
        path = self._paths.ranking_path(topic_slug, filename)
        await self._write_text_async(path, content)

    async def save_aggregation_report(self, filename: str, content: str) -> None:
        """Save a cross-topic aggregation report file."""
        path = self._paths.aggregation_path(filename)
        await self._write_text_async(path, content)

    async def save_ranking_output(
        self, topic_slug: str, leaderboard: list[Any], _ranking_system: Any
    ) -> None:
        """Save leaderboard to text/csv files for human consumption."""

        def _save() -> None:
            md_path = self._paths.ranking_path(topic_slug, "leaderboard.md")
            with md_path.open("w", encoding="utf-8") as f:
                f.write(f"# Leaderboard: {topic_slug}\n\n")
                f.write("| Rank | Candidate | Rating | Matches | Wins | Losses |\n")
                f.write("|---|---|---|---|---|---|\n")
                for i, r in enumerate(leaderboard, 1):
                    f.write(
                        f"| {i} | {r.candidate_id} | {r.rating:.1f} | "
                        f"{r.matches} | {r.wins} | {r.losses} |\n"
                    )

            csv_path = self._paths.ranking_path(topic_slug, "leaderboard.csv")
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
            json_path = self._paths.ranking_path(topic_slug, "leaderboard.json")
            data = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in leaderboard]
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        await asyncio.to_thread(_save)
