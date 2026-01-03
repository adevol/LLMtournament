"""Database storage for matches and ratings using SQLModel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from sqlmodel import Session, col, select

from llm_tournament.models import Match, Rating

if TYPE_CHECKING:
    from sqlalchemy import Engine

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


class DBStoreMixin:
    """Mixin for SQLModel-based match and rating storage."""

    _engine: Engine
    base_dir: Path

    def ranking_dir(self, topic_slug: str) -> Path:
        """Get or create ranking directory (protocol for mixin compatibility)."""
        raise NotImplementedError

    async def save_match(self, topic_slug: str, match_data: Any) -> None:
        """Save a match result to database and JSONL backup."""
        data = (
            match_data.model_dump()
            if hasattr(match_data, "model_dump")
            else dict(match_data)
        )

        if "topic_slug" not in data:
            data["topic_slug"] = topic_slug

        data["reasons"] = _normalize_reasons(data.get("reasons"))

        def _save() -> None:
            with Session(self._engine) as session:
                match = Match.model_validate(data)
                session.add(match)
                session.commit()

            ranking_dir = self.ranking_dir(topic_slug)
            jsonl_path = ranking_dir / "matches.jsonl"
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")

        await asyncio.to_thread(_save)

    async def get_matches_for_essay(
        self, topic_slug: str, essay_id: str
    ) -> list[dict[str, Any]]:
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
        data = (
            rating_data.model_dump()
            if hasattr(rating_data, "model_dump")
            else dict(rating_data)
        )

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
        """Get all ratings across all topics.

        Returns:
            List of Rating objects.
        """

        def _get() -> list[Any]:
            with Session(self._engine) as session:
                statement = select(Rating)
                results = session.exec(statement).all()
                return list(results)

        return await asyncio.to_thread(_get)
