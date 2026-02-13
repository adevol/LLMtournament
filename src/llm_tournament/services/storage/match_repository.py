"""Database persistence for match records."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, select

from llm_tournament.models import Match

from .paths import StoragePaths
from .repository import AsyncRepository

if TYPE_CHECKING:
    from sqlalchemy import Engine


def _normalize_reasons(reasons: Any) -> list[str]:
    """Normalize reasons field to a list of strings."""
    if isinstance(reasons, str):
        return [reasons]
    if reasons is None:
        return []
    if isinstance(reasons, list):
        return [str(r) for r in reasons]
    return [str(reasons)]


class MatchRepository(AsyncRepository):
    """Persist and query match records."""

    def __init__(self, engine: Engine, paths: StoragePaths) -> None:
        super().__init__(engine)
        self._paths = paths

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

        await self._run_session(_db_save)

        def _save_jsonl() -> None:
            jsonl_path = self._paths.ranking_path(topic_slug, "matches.jsonl")
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

        return await self._run_session(_get)
