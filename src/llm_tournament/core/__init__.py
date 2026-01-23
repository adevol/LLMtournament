"""Core configuration and utilities for LLM Tournament."""

from llm_tournament.core.config import (
    DEFAULT_SLUG_MAX_LENGTH,
    RankingConfig,
    TopicConfig,
    TournamentConfig,
    WriterConfig,
    calculate_nr_rounds,
    hash_messages,
    load_config,
)
from llm_tournament.core.errors import (
    APIKeyError,
    ConfigurationError,
    EmptyModelError,
    MissingFieldError,
    ValidationError,
)
from llm_tournament.core.progress import TournamentProgress
from llm_tournament.core.slug import SlugGenerator

__all__ = [
    "DEFAULT_SLUG_MAX_LENGTH",
    "RankingConfig",
    "TopicConfig",
    "TournamentConfig",
    "WriterConfig",
    "SlugGenerator",
    "TournamentProgress",
    "calculate_nr_rounds",
    "hash_messages",
    "load_config",
    "APIKeyError",
    "ConfigurationError",
    "EmptyModelError",
    "MissingFieldError",
    "ValidationError",
]
