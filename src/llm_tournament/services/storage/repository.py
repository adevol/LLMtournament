"""Shared async repository helpers for SQLModel session work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from sqlmodel import Session

if TYPE_CHECKING:
    from sqlalchemy import Engine

T = TypeVar("T")


class AsyncRepository(Generic[T]):
    """Wrap sync SQLModel session work for async callers."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    async def _run_session(self, fn: Callable[[Session], T]) -> T:
        """Run a sync function inside a Session on a worker thread."""

        def _run() -> T:
            with Session(self._engine) as session:
                return fn(session)

        return await asyncio.to_thread(_run)
