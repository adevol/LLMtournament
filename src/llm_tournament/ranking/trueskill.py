"""TrueSkill/OpenSkill ranking system for LLM Tournament."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from openskill.models import PlackettLuce


@dataclass
class TrueSkillRating:
    """Tracks TrueSkill rating for a candidate.

    Attributes:
        mu: Mean skill estimate.
        sigma: Uncertainty in skill estimate.
        matches: Number of matches played.
        wins: Number of wins.
        losses: Number of losses.
    """

    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def ordinal(self) -> float:
        """Conservative skill estimate (mu - 3*sigma).

        Returns:
            Ordinal rating for ranking purposes.
        """
        return self.mu - 3 * self.sigma

    def record_match(self, new_mu: float, new_sigma: float, won: bool) -> None:
        """Record a match result.

        Args:
            new_mu: New mu after match.
            new_sigma: New sigma after match.
            won: Whether this candidate won.
        """
        self.history.append((self.mu, self.sigma))
        self.mu = new_mu
        self.sigma = new_sigma
        self.matches += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1


class TrueSkillSystem:
    """TrueSkill/OpenSkill ranking system implementing the RankingSystem protocol.

    Uses the PlackettLuce model from openskill for Bayesian skill estimation.
    Converges faster than Elo and handles uncertainty explicitly.

    Attributes:
        initial_mu: Starting mean skill estimate.
        initial_sigma: Starting uncertainty.
    """

    def __init__(
        self,
        initial_mu: float = 25.0,
        initial_sigma: float | None = None,
    ) -> None:
        """Initialize TrueSkill system.

        Args:
            initial_mu: Starting mean skill estimate.
            initial_sigma: Starting uncertainty (default: mu/3).
        """
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma if initial_sigma else initial_mu / 3.0
        self._ratings: dict[str, TrueSkillRating] = {}
        self._model = PlackettLuce()

    def initialize(self, candidate_ids: Sequence[str]) -> None:
        """Initialize ratings for all candidates.

        Args:
            candidate_ids: List of unique candidate identifiers.
        """
        self._ratings = {
            cid: TrueSkillRating(mu=self.initial_mu, sigma=self.initial_sigma)
            for cid in candidate_ids
        }

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
            confidence: Judge confidence (0.0-1.0). Higher confidence
                reduces the weight of uncertainty in the update.

        Returns:
            Tuple of (new_winner_ordinal, new_loser_ordinal).
        """
        winner = self._ratings[winner_id]
        loser = self._ratings[loser_id]

        # Create rating objects for openskill
        winner_rating = self._model.rating(mu=winner.mu, sigma=winner.sigma)
        loser_rating = self._model.rating(mu=loser.mu, sigma=loser.sigma)

        # Rate the match (winner first = rank 1, loser second = rank 2)
        # Low confidence -> larger sigma adjustment (more uncertainty)
        # We scale sigma based on confidence before rating
        if confidence < 1.0:
            # Increase sigma temporarily to reduce rating change impact
            sigma_scale = 1.0 + (1.0 - confidence) * 0.5
            loser_rating = self._model.rating(mu=loser.mu, sigma=loser.sigma * sigma_scale)
            winner_rating = self._model.rating(mu=winner.mu, sigma=winner.sigma * sigma_scale)

        # Rate the match: [[winner], [loser]] with ranks [1, 2]
        new_ratings = self._model.rate([[winner_rating], [loser_rating]])

        new_winner_rating = new_ratings[0][0]
        new_loser_rating = new_ratings[1][0]

        # Record the match
        winner.record_match(new_winner_rating.mu, new_winner_rating.sigma, won=True)
        loser.record_match(new_loser_rating.mu, new_loser_rating.sigma, won=False)

        return winner.ordinal, loser.ordinal

    def get_rating(self, candidate_id: str) -> float:
        """Get current ordinal rating for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Ordinal rating.
        """
        return self._ratings[candidate_id].ordinal

    def get_stats(self, candidate_id: str) -> dict[str, int]:
        """Get match statistics for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Dict with 'matches', 'wins', 'losses' keys.
        """
        r = self._ratings[candidate_id]
        return {"matches": r.matches, "wins": r.wins, "losses": r.losses}

    def get_mu_sigma(self, candidate_id: str) -> tuple[float, float]:
        """Get mu and sigma for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Tuple of (mu, sigma).
        """
        r = self._ratings[candidate_id]
        return r.mu, r.sigma

    def get_leaderboard(self) -> list[tuple[str, float, int, int, float, float]]:
        """Get sorted leaderboard with extended TrueSkill stats.

        Returns:
            List of (candidate_id, ordinal, wins, losses, mu, sigma) tuples,
            sorted by ordinal descending.
        """
        entries = [
            (cid, r.ordinal, r.wins, r.losses, r.mu, r.sigma) for cid, r in self._ratings.items()
        ]
        return sorted(entries, key=lambda x: x[1], reverse=True)
