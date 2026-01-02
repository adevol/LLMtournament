"""Ranking module for LLM Tournament.

Provides pluggable ranking systems (Elo, TrueSkill) and match judging logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_tournament.ranking.base import RankingSystem
from llm_tournament.ranking.elo import (
    EloRating,
    EloSystem,
    calculate_expected_win_chance,
    update_elo,
)
from llm_tournament.ranking.trueskill import TrueSkillRating, TrueSkillSystem

if TYPE_CHECKING:
    from llm_tournament.core.config import TournamentConfig


def create_ranking_system(config: TournamentConfig) -> RankingSystem:
    """Create ranking system based on config.

    Args:
        config: Tournament configuration.

    Returns:
        Configured ranking system.
    """
    if config.ranking.algorithm == "trueskill":
        return TrueSkillSystem(
            initial_mu=config.ranking.initial_mu,
            initial_sigma=config.ranking.initial_sigma,
        )
    # Default to Elo
    return EloSystem(
        initial_rating=config.ranking.initial_elo,
        k_factor=config.ranking.k_factor,
    )


__all__ = [
    "EloRating",
    "EloSystem",
    "RankingSystem",
    "TrueSkillRating",
    "TrueSkillSystem",
    "calculate_expected_win_chance",
    "create_ranking_system",
    "update_elo",
]
