"""Base protocol for ranking systems in LLM Tournament."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class RankingSystem(Protocol):
    """Protocol for ranking algorithms.

    Implementations must provide methods for updating ratings after matches
    and retrieving current standings.
    """

    def initialize(self, candidate_ids: Sequence[str]) -> None:
        """Initialize ratings for all candidates.

        Args:
            candidate_ids: List of unique candidate identifiers.
        """
        ...

    def update(
        self,
        winner_id: str,
        loser_id: str,
        confidence: float = 1.0,
    ) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            winner_id: ID of the winning candidate.
            loser_id: ID of the losing candidate.
            confidence: Judge confidence score (0.0-1.0).

        Returns:
            Tuple of (new_winner_rating, new_loser_rating).
        """
        ...

    def get_rating(self, candidate_id: str) -> float:
        """Get current rating for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Current rating value.
        """
        ...

    def get_stats(self, candidate_id: str) -> dict[str, int]:
        """Get match statistics for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Dict with 'matches', 'wins', 'losses' keys.
        """
        ...

    def get_leaderboard(self) -> list[tuple[str, float, int, int]]:
        """Get sorted leaderboard.

        Returns:
            List of (candidate_id, rating, wins, losses) tuples,
            sorted by rating descending.
        """
        ...
