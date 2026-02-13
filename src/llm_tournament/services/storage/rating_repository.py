"""Database persistence for rating records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlmodel import Session, col, select

from llm_tournament.models import Rating

from .repository import AsyncRepository

if TYPE_CHECKING:
    from sqlalchemy import Engine


class RatingRepository(AsyncRepository):
    """Persist and query rating records."""

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)

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

        await self._run_session(_save)

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

        return await self._run_session(_get)

    async def get_all_ratings(self) -> list[Any]:
        """Get all ratings across all topics."""

        def _get(session: Session) -> list[Any]:
            statement = select(Rating)
            results = session.exec(statement).all()
            return list(results)

        return await self._run_session(_get)
