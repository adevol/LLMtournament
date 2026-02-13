"""Path utilities for tournament storage artifacts."""

from __future__ import annotations

from pathlib import Path


class StoragePaths:
    """Build and create filesystem paths used by storage services."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def topic_dir(self, topic_slug: str) -> Path:
        """Get or create topic directory."""
        path = self.base_dir / topic_slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def topic_subdir(self, topic_slug: str, subdir: str) -> Path:
        """Get or create a topic subdirectory."""
        path = self.topic_dir(topic_slug) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def essay_path(self, topic_slug: str, essay_id: str, version: str) -> Path:
        """Build path to an essay markdown file."""
        subdir = "v0" if version == "v0" else "v1"
        return self.topic_subdir(topic_slug, subdir) / f"{essay_id}.md"

    def feedback_path(self, topic_slug: str, writer_slug: str, critic_slug: str) -> Path:
        """Build path to a feedback markdown file."""
        return self.topic_subdir(topic_slug, "feedback") / f"{writer_slug}__{critic_slug}.md"

    def revision_path(self, topic_slug: str, writer_slug: str, critic_slug: str) -> Path:
        """Build path to a revised essay markdown file."""
        return self.topic_subdir(topic_slug, "v1") / f"{writer_slug}__{critic_slug}.md"

    def ranking_path(self, topic_slug: str, filename: str) -> Path:
        """Build path to a ranking artifact file."""
        return self.topic_subdir(topic_slug, "ranking") / filename

    def aggregation_path(self, filename: str) -> Path:
        """Build path to a cross-topic aggregation artifact."""
        output_dir = self.base_dir / "final_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        path = output_dir / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        return path
