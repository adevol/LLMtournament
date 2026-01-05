"""Elo rating calculations for LLM Tournament."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class EloRating:
    """Tracks Elo rating for a candidate.

    Attributes:
        rating: Current Elo rating.
        matches: Number of matches played.
        wins: Number of wins.
        losses: Number of losses.
    """

    rating: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    history: list[float] = field(default_factory=list)

    def record_match(self, new_rating: float, won: bool) -> None:
        """Record a match result.

        Args:
            new_rating: New Elo rating after match.
            won: Whether this candidate won.
        """
        self.history.append(self.rating)
        self.rating = new_rating
        self.matches += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1


def calculate_expected_win_chance(rating_a: float, rating_b: float) -> float:
    """Calculate expected win probability for player A against player B.

    Uses the standard Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Rating of player A.
        rating_b: Rating of player B.

    Returns:
        Probability that A wins (0.0 to 1.0).
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    winner: str,
    k_factor: float = 32.0,
    confidence: float = 1.0,
) -> tuple[float, float]:
    """Update Elo ratings after a match.

    Args:
        rating_a: Current rating of candidate A.
        rating_b: Current rating of candidate B.
        winner: "A" or "B" indicating the winner.
        k_factor: Base K-factor for updates.
        confidence: Judge confidence (0.0-1.0) for weighted K.

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    # Calculate expected scores
    expected_a = calculate_expected_win_chance(rating_a, rating_b)
    expected_b = 1.0 - expected_a

    # Actual scores (1 for win, 0 for loss)
    if winner == "A":
        actual_a, actual_b = 1.0, 0.0
    else:
        actual_a, actual_b = 0.0, 1.0

    # Confidence-weighted K-factor
    # effective_K = K * (0.5 + confidence / 2)
    # At confidence=0.0 -> K * 0.5 (half weight)
    # At confidence=1.0 -> K * 1.0 (full weight)
    effective_k = k_factor * (0.5 + confidence / 2)

    # Update ratings
    new_rating_a = rating_a + effective_k * (actual_a - expected_a)
    new_rating_b = rating_b + effective_k * (actual_b - expected_b)

    return new_rating_a, new_rating_b


class EloSystem:
    """Elo ranking system implementing the RankingSystem protocol.

    Attributes:
        initial_rating: Starting Elo rating for new candidates.
        k_factor: Base K-factor for rating adjustments.
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
    ) -> None:
        """Initialize Elo system.

        Args:
            initial_rating: Starting rating for candidates.
            k_factor: Base K-factor for rating adjustments.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: dict[str, EloRating] = {}

    def initialize(self, candidate_ids: Sequence[str]) -> None:
        """Initialize ratings for all candidates.

        Args:
            candidate_ids: List of unique candidate identifiers.
        """
        self._ratings = {cid: EloRating(rating=self.initial_rating) for cid in candidate_ids}

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
        winner_rating = self._ratings[winner_id].rating
        loser_rating = self._ratings[loser_id].rating

        new_winner, new_loser = update_elo(
            winner_rating,
            loser_rating,
            winner="A",
            k_factor=self.k_factor,
            confidence=confidence,
        )

        self._ratings[winner_id].record_match(new_winner, won=True)
        self._ratings[loser_id].record_match(new_loser, won=False)

        return new_winner, new_loser

    def get_rating(self, candidate_id: str) -> float:
        """Get current rating for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Current Elo rating.
        """
        return self._ratings[candidate_id].rating

    def get_stats(self, candidate_id: str) -> dict[str, int]:
        """Get match statistics for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Dict with 'matches', 'wins', 'losses' keys.
        """
        r = self._ratings[candidate_id]
        return {"matches": r.matches, "wins": r.wins, "losses": r.losses}

    def get_leaderboard(self) -> list[tuple[str, float, int, int]]:
        """Get sorted leaderboard.

        Returns:
            List of (candidate_id, rating, wins, losses) tuples,
            sorted by rating descending.
        """
        entries = [(cid, r.rating, r.wins, r.losses) for cid, r in self._ratings.items()]
        return sorted(entries, key=lambda x: x[1], reverse=True)

    def get_rating_object(self, candidate_id: str) -> EloRating:
        """Get the full EloRating object for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            EloRating dataclass instance.
        """
        return self._ratings[candidate_id]
