"""File-backed storage for essays and feedback artifacts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from .paths import StoragePaths

logger = structlog.get_logger()


class EssayStore:
    """Persist and load essay-related files."""

    def __init__(self, paths: StoragePaths) -> None:
        self._paths = paths

    @staticmethod
    def _write_text(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    async def _write_text_async(self, path: Path, content: str) -> Path:
        return await asyncio.to_thread(self._write_text, path, content)

    async def _read_text_async(self, path: Path) -> str:
        return await asyncio.to_thread(self._read_text, path)

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> Path:
        """Save an essay to file."""
        path = self._paths.essay_path(topic_slug, writer_slug, version)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_essay", path=str(saved_path))
        return saved_path

    async def load_essay(self, topic_slug: str, essay_id: str, version: str) -> str:
        """Load an essay from file."""
        path = self._paths.essay_path(topic_slug, essay_id, version)
        return await self._read_text_async(path)

    async def save_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save feedback to file."""
        path = self._paths.feedback_path(topic_slug, writer_slug, critic_slug)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_feedback", path=str(saved_path))
        return saved_path

    async def load_feedback(self, topic_slug: str, writer_slug: str, critic_slug: str) -> str:
        """Load feedback from file."""
        path = self._paths.feedback_path(topic_slug, writer_slug, critic_slug)
        return await self._read_text_async(path)

    async def save_revision(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save revised essay to file."""
        path = self._paths.revision_path(topic_slug, writer_slug, critic_slug)
        saved_path = await self._write_text_async(path, content)
        logger.debug("saved_revision", path=str(saved_path))
        return saved_path
