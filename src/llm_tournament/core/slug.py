"""Slug generation utilities for filesystem-safe identifiers."""

from __future__ import annotations

import hashlib
import re


class SlugGenerator:
    """Generate filesystem-safe slugs from arbitrary inputs.

    Provides a unified interface for creating consistent, safe identifiers
    across the codebase, replacing multiple scattered slug functions.
    """

    def __init__(self, max_length: int | None = 50) -> None:
        """Initialize slug generator.

        Args:
            max_length: Maximum length for generated slugs. Use None to disable truncation.
        """
        self.max_length = max_length

    def slugify(self, value: str) -> str:
        """Generate a URL-safe slug from free text."""
        return self.truncate(self._slugify(value))

    def safe_slug(
        self,
        value: str,
        *,
        hash_content: str | None = None,
        suffix: str | None = None,
    ) -> str:
        """Generate a filesystem-safe ID with optional hash/suffix."""
        slug = self.safe_id(value)
        if hash_content is not None:
            hash_suffix = hashlib.sha256(hash_content.encode()).hexdigest()[:6]
            slug = f"{slug}_{hash_suffix}"
        if suffix:
            slug = f"{slug}_{self.safe_id(suffix)}"
        return self.truncate(slug)

    def truncate(self, value: str) -> str:
        """Truncate a value to the configured max length."""
        if self.max_length is None:
            return value
        return value[: self.max_length]

    @staticmethod
    def _slugify(value: str) -> str:
        """Generate a URL-safe slug from free text."""
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    @staticmethod
    def safe_id(value: str) -> str:
        """Convert arbitrary ID to filesystem-safe format.

        Args:
            value: The ID to convert.

        Returns:
            A filesystem-safe string.
        """
        return value.replace("/", "__").replace(":", "_")
