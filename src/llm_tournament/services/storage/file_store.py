"""File-based storage for essays, feedback, and revisions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

logger = structlog.get_logger()


class FileStoreMixin:
    """Mixin for file-based essay/feedback/revision storage."""

    base_dir: Path

    def topic_dir(self, topic_slug: str) -> Path:
        """Get or create topic directory."""
        path = self.base_dir / topic_slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def v0_dir(self, topic_slug: str) -> Path:
        """Get or create v0 essays directory."""
        path = self.topic_dir(topic_slug) / "v0"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def feedback_dir(self, topic_slug: str) -> Path:
        """Get or create feedback directory."""
        path = self.topic_dir(topic_slug) / "feedback"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def v1_dir(self, topic_slug: str) -> Path:
        """Get or create v1 essays directory."""
        path = self.topic_dir(topic_slug) / "v1"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ranking_dir(self, topic_slug: str) -> Path:
        """Get or create ranking directory."""
        path = self.topic_dir(topic_slug) / "ranking"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> Path:
        """Save an essay to file."""

        def _save() -> Path:
            directory = (
                self.v0_dir(topic_slug) if version == "v0" else self.v1_dir(topic_slug)
            )
            path = directory / f"{writer_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_essay", path=str(path))
            return path

        return await asyncio.to_thread(_save)

    async def load_essay(self, topic_slug: str, essay_id: str, version: str) -> str:
        """Load an essay from file."""

        def _load() -> str:
            directory = (
                self.v0_dir(topic_slug) if version == "v0" else self.v1_dir(topic_slug)
            )
            path = directory / f"{essay_id}.md"
            return path.read_text(encoding="utf-8")

        return await asyncio.to_thread(_load)

    async def save_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save feedback to file."""

        def _save() -> Path:
            path = self.feedback_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_feedback", path=str(path))
            return path

        return await asyncio.to_thread(_save)

    async def load_feedback(
        self, topic_slug: str, writer_slug: str, critic_slug: str
    ) -> str:
        """Load feedback from file."""

        def _load() -> str:
            path = self.feedback_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            return path.read_text(encoding="utf-8")

        return await asyncio.to_thread(_load)

    async def save_revision(
        self, topic_slug: str, writer_slug: str, critic_slug: str, content: str
    ) -> Path:
        """Save revised essay to file."""

        def _save() -> Path:
            path = self.v1_dir(topic_slug) / f"{writer_slug}__{critic_slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.debug("saved_revision", path=str(path))
            return path

        return await asyncio.to_thread(_save)
