"""Configuration schemas and loading for LLM Tournament."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_SLUG_MAX_LENGTH = 50


def safe_id(value: str) -> str:
    """Convert arbitrary ID to filesystem-safe slug."""
    return value.replace("/", "__").replace(":", "_")


def truncate_slug(value: str, max_length: int) -> str:
    """Cap slug length to keep filesystem paths manageable on systems with path length
    limits (e.g., Windows)."""
    if len(value) <= max_length:
        return value
    return value[:max_length]


class TopicConfig(BaseModel):
    """Configuration for a single essay topic."""

    title: str
    prompts: dict[str, str] = Field(default_factory=dict)
    source_pack: str | None = None
    slug_max_length: int | None = Field(default=None, ge=1)

    @field_validator("prompts")
    @classmethod
    def validate_prompts_not_empty(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("At least one prompt must be defined in 'prompts'")
        return v

    @property
    def slug(self) -> str:
        """Generate URL-safe slug from title."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")
        max_length = self.slug_max_length or DEFAULT_SLUG_MAX_LENGTH
        return truncate_slug(slug, max_length)


class WriterConfig(BaseModel):
    """Configuration for a writer participant with optional custom system prompt.

    Allows comparing different system prompts for the same model in tournaments.

    Attributes:
        model_id: The model identifier (e.g., "openai/gpt-4").
        system_prompt: Optional custom system prompt. If None, uses the default.
        name: Optional display name. If None, auto-generated from model_id + prompt hash.
    """

    model_id: str
    system_prompt: str | None = None
    name: str | None = None

    def get_slug(self, max_length: int = DEFAULT_SLUG_MAX_LENGTH) -> str:
        """Generate a unique slug for this writer config.

        If name is provided, uses that. Otherwise, generates from model_id
        plus a short hash of the system_prompt (if custom).
        """
        if self.name:
            base = safe_id(self.name)
        else:
            base = safe_id(self.model_id)
            if self.system_prompt:
                # Append short hash for uniqueness
                prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:6]
                base = f"{base}_{prompt_hash}"
        return truncate_slug(base, max_length)


class RankingConfig(BaseModel):
    """Ranking algorithm configuration.

    Attributes:
        algorithm: Rating algorithm ("elo" or "trueskill").
        judging_method: How judges evaluate matches:
            - "audit": Single primary judge, audits on low confidence.
            - "parallel_majority": N judges in parallel, majority vote.
        rounds: Number of Swiss tournament rounds. If None, auto-calculated as
            ceil(log2(candidates)) + 1, which provides stable rankings.
        audit_confidence_threshold: Confidence below which to trigger audit/expansion.
        primary_judges: Judges for primary voting. Defaults to main judges list.
        sub_judges: Backup judges for low-confidence expansion.
        primary_judge_count: How many primary judges to use (default 3).
        sub_judge_count: How many sub-judges to add on low confidence (default 2).
    """

    algorithm: Literal["elo", "trueskill"] = "elo"
    judging_method: Literal["audit", "parallel_majority"] = "audit"
    rounds: int | None = None  # Auto-calculated if None: ceil(log2(N)) + 1
    audit_confidence_threshold: float = 0.7
    primary_judges: list[str] | None = None
    sub_judges: list[str] | None = None
    primary_judge_count: int = 3
    sub_judge_count: int = 2
    # Elo-specific
    initial_elo: float = 1500.0
    k_factor: float = 32.0
    # TrueSkill-specific
    initial_mu: float = 25.0
    initial_sigma: float | None = None  # Default value if None: mu / 3


class TournamentConfig(BaseModel):
    """Complete tournament configuration."""

    # Model lists - writers can be strings or WriterConfig objects
    writers: list[str | WriterConfig] = Field(..., min_length=1)
    critics: list[str] = Field(..., min_length=1)
    judges: list[str] = Field(..., min_length=1)
    topics: list[TopicConfig] = Field(..., min_length=1)

    # Token caps per role
    writer_tokens: int = 1200
    critic_tokens: int = 300
    revision_tokens: int = 1300
    judge_tokens: int = 500

    # Temperature settings per role
    writer_temp: float = 0.7
    critic_temp: float = 0.3
    revision_temp: float = 0.5
    judge_temp: float = 0.2

    # Analysis settings
    analysis_top_k: int = 10

    # Ranking configuration (kept as nested - complex enough)
    ranking: RankingConfig = Field(default_factory=RankingConfig)

    # Custom system prompts (None = use defaults from prompts.yaml)
    writer_system_prompt: str | None = None
    judge_system_prompt: str | None = None

    # Other settings
    simple_mode: bool = False
    seed: int = 42
    slug_max_length: int = Field(default=50, ge=1, le=100)
    output_dir: str = "./runs"
    api_key: str | None = None

    @field_validator("critics", "judges")
    @classmethod
    def validate_model_list(cls, v: list[str]) -> list[str]:
        """Ensure model IDs are non-empty strings."""
        for model_id in v:
            if not model_id or not model_id.strip():
                raise ValueError("Model IDs cannot be empty")
        return v

    @field_validator("writers")
    @classmethod
    def validate_writers_list(cls, v: list[str | WriterConfig]) -> list[str | WriterConfig]:
        """Ensure writer entries are valid (strings or WriterConfig)."""
        for writer in v:
            if isinstance(writer, str) and (not writer or not writer.strip()):
                raise ValueError("Writer model IDs cannot be empty")
            if isinstance(writer, WriterConfig) and (
                not writer.model_id or not writer.model_id.strip()
            ):
                raise ValueError("WriterConfig.model_id cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        for topic in self.topics:
            if topic.slug_max_length is None:
                topic.slug_max_length = self.slug_max_length

    def get_slug_model(self, model_id: str, max_length: int | None = None) -> str:
        """Convert model ID to filesystem-safe slug."""
        if max_length is None:
            max_length = self.slug_max_length
        return truncate_slug(safe_id(model_id), max_length)

    def get_writer_slug(self, writer: str | WriterConfig) -> str:
        """Get slug for a writer (string or WriterConfig)."""
        if isinstance(writer, WriterConfig):
            return writer.get_slug(self.slug_max_length)
        return self.get_slug_model(writer)

    def get_writer_model_id(self, writer: str | WriterConfig) -> str:
        """Get model ID for a writer (string or WriterConfig)."""
        if isinstance(writer, WriterConfig):
            return writer.model_id
        return writer

    def get_writer_system_prompt(self, writer: str | WriterConfig) -> str | None:
        """Get custom system prompt for a writer, or None for default."""
        if isinstance(writer, WriterConfig):
            return writer.system_prompt
        return None

    def get_slug_topic(self, title: str, max_length: int | None = None) -> str:
        """Convert a topic title to a URL-safe slug."""
        if max_length is None:
            max_length = self.slug_max_length
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        return truncate_slug(slug, max_length)

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            msg = (
                "API key required. Set OPENROUTER_API_KEY env var "
                "or api_key in config (avoid committing secrets to git)."
            )
            raise ValueError(msg)
        return key


def load_config(path: str | Path) -> TournamentConfig:
    """Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Validated TournamentConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        data = yaml.safe_load(f)

    return TournamentConfig.model_validate(data)


def calculate_nr_rounds(num_candidates: int) -> int:
    """Calculate recommended Swiss tournament rounds.

    Uses ceil(log2(N)) + 1 heuristic to ensure stable rankings:
    - log2(N) rounds needed to find a clear winner
    - +1 extra round for ranking stability

    Args:
        num_candidates: Number of participants in the tournament.

    Returns:
        Recommended number of rounds (minimum 3).
    """
    if num_candidates <= 1:
        return 1
    min_rounds = 3
    return max(min_rounds, math.ceil(math.log2(num_candidates)) + 1)


def hash_messages(messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    """Create deterministic hash for cache key.

    Args:
        messages: List of message dicts with role/content.
        params: Additional parameters (model, temperature, etc.).

    Returns:
        SHA-256 hash as hex string.
    """
    content = json.dumps({"messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
