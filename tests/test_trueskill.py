"""Tests for TrueSkill ranking system."""

import pytest

from llm_tournament.ranking.trueskill import TrueSkillRating, TrueSkillSystem


class TestTrueSkillRating:
    """Tests for TrueSkillRating dataclass."""

    def test_defaults(self):
        """Test default values."""
        rating = TrueSkillRating()
        assert rating.mu == 25.0
        assert rating.sigma == pytest.approx(25.0 / 3.0, abs=0.01)
        assert rating.matches == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.history == []

    def test_ordinal(self):
        """Test ordinal calculation (mu - 3*sigma)."""
        rating = TrueSkillRating(mu=25.0, sigma=25.0 / 3.0)
        # ordinal = 25 - 3 * (25/3) = 25 - 25 = 0
        assert rating.ordinal == pytest.approx(0.0, abs=0.01)

    def test_record_win(self):
        """Test recording a win."""
        rating = TrueSkillRating()
        rating.record_match(new_mu=27.0, new_sigma=7.5, won=True)

        assert rating.mu == 27.0
        assert rating.sigma == 7.5
        assert rating.matches == 1
        assert rating.wins == 1
        assert rating.losses == 0
        assert len(rating.history) == 1

    def test_record_loss(self):
        """Test recording a loss."""
        rating = TrueSkillRating()
        rating.record_match(new_mu=23.0, new_sigma=7.5, won=False)

        assert rating.mu == 23.0
        assert rating.sigma == 7.5
        assert rating.matches == 1
        assert rating.wins == 0
        assert rating.losses == 1


class TestTrueSkillSystem:
    """Tests for TrueSkillSystem ranking."""

    def test_initialize(self):
        """Test initialization of candidates."""
        system = TrueSkillSystem()
        system.initialize(["a", "b", "c"])

        assert system.get_rating("a") == pytest.approx(0.0, abs=0.1)
        assert system.get_rating("b") == pytest.approx(0.0, abs=0.1)
        assert system.get_rating("c") == pytest.approx(0.0, abs=0.1)

    def test_update_winner_gains(self):
        """Test that winner gains rating after match."""
        system = TrueSkillSystem()
        system.initialize(["a", "b"])

        initial_a = system.get_rating("a")
        initial_b = system.get_rating("b")

        system.update("a", "b", confidence=1.0)

        assert system.get_rating("a") > initial_a
        assert system.get_rating("b") < initial_b

    def test_stats_tracking(self):
        """Test match statistics are tracked."""
        system = TrueSkillSystem()
        system.initialize(["a", "b"])

        system.update("a", "b", confidence=1.0)

        stats_a = system.get_stats("a")
        assert stats_a["matches"] == 1
        assert stats_a["wins"] == 1
        assert stats_a["losses"] == 0

        stats_b = system.get_stats("b")
        assert stats_b["matches"] == 1
        assert stats_b["wins"] == 0
        assert stats_b["losses"] == 1

    def test_leaderboard_sorted(self):
        """Test leaderboard is sorted by rating descending."""
        system = TrueSkillSystem()
        system.initialize(["a", "b", "c"])

        # a beats b, b beats c
        system.update("a", "b", confidence=1.0)
        system.update("b", "c", confidence=1.0)

        leaderboard = system.get_leaderboard()

        # Should be sorted: a > b > c
        assert leaderboard[0][0] == "a"
        assert leaderboard[1][0] == "b"
        assert leaderboard[2][0] == "c"

    def test_mu_sigma_access(self):
        """Test accessing raw mu and sigma."""
        system = TrueSkillSystem(initial_mu=30.0, initial_sigma=10.0)
        system.initialize(["a"])

        mu, sigma = system.get_mu_sigma("a")
        assert mu == 30.0
        assert sigma == 10.0

    def test_confidence_affects_update(self):
        """Test that confidence parameter works without errors."""
        system = TrueSkillSystem()
        system.initialize(["a", "b"])

        # Should work with various confidence values
        system.update("a", "b", confidence=1.0)
        system.update("a", "b", confidence=0.5)
        system.update("a", "b", confidence=0.0)

        # Check that ratings were updated
        stats = system.get_stats("a")
        assert stats["matches"] == 3
        assert stats["wins"] == 3
