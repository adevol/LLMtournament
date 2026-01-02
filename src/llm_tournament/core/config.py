"""Configuration schemas and loading for LLM Tournament."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class TopicConfig(BaseModel):
    """Configuration for a single essay topic."""

    title: str
    prompts: dict[str, str] = Field(default_factory=dict)
    source_pack: str | None = None

    @field_validator("prompts")
    @classmethod
    def validate_prompts_not_empty(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("At least one prompt must be defined in 'prompts'")
        return v

    @property
    def slug(self) -> str:
        """Generate URL-safe slug from title."""
        return re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")


class TokenCaps(BaseModel):
    """Token limits per role."""

    writer_tokens: int = 1200
    critic_tokens: int = 300
    revision_tokens: int = 1300
    judge_tokens: int = 500


class Temperatures(BaseModel):
    """Temperature settings per role."""

    writer: float = 0.7
    critic: float = 0.3
    revision: float = 0.5
    judge: float = 0.2


class RankingConfig(BaseModel):
    """Ranking algorithm configuration."""

    algorithm: Literal["elo", "trueskill"] = "elo"
    rounds: int = 5
    audit_confidence_threshold: float = 0.7
    # Elo-specific
    initial_elo: float = 1500.0
    k_factor: float = 32.0
    # TrueSkill-specific
    initial_mu: float = 25.0
    initial_sigma: float | None = None  # Default value if None: mu / 3


class AnalysisConfig(BaseModel):
    """Post-ranking analysis configuration."""

    top_k: int = 10


class TournamentConfig(BaseModel):
    """Complete tournament configuration."""

    writers: list[str] = Field(..., min_length=1)
    critics: list[str] = Field(..., min_length=1)
    judges: list[str] = Field(..., min_length=1)
    topics: list[TopicConfig] = Field(..., min_length=1)
    token_caps: TokenCaps = Field(default_factory=TokenCaps)
    temperatures: Temperatures = Field(default_factory=Temperatures)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    simple_mode: bool = False
    seed: int = 42
    output_dir: str = "./runs"
    api_key: str | None = None

    @field_validator("writers", "critics", "judges")
    @classmethod
    def validate_model_list(cls, v: list[str]) -> list[str]:
        """Ensure model IDs are non-empty strings."""
        for model_id in v:
            if not model_id or not model_id.strip():
                msg = "Model IDs cannot be empty"
                raise ValueError(msg)
        return v

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


def model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug.

    Args:
        model_id: OpenRouter model ID (e.g., 'openai/gpt-4-turbo').

    Returns:
        Filesystem-safe slug (e.g., 'openai__gpt-4-turbo').
    """
    return model_id.replace("/", "__").replace(":", "_")


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


def get_git_hash() -> str | None:
    """Get current git commit hash if available.

    Returns:
        Short git hash or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
