"""Tests for Elo rating calculations."""

import pytest

from llm_tournament.ranking.elo import (
    EloRating,
    EloSystem,
    calculate_expected_win_chance,
    update_elo,
)


class TestCalculateExpectedWinChance:
    """Tests for expected win probability calculation."""

    def test_equal_ratings(self):
        """Test equal ratings produce 0.5 expected."""
        expected = calculate_expected_win_chance(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.001)

    def test_higher_rating_higher_expected(self):
        """Test higher rated player has higher expected score."""
        expected = calculate_expected_win_chance(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    def test_lower_rating_lower_expected(self):
        """Test lower rated player has lower expected score."""
        expected = calculate_expected_win_chance(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0

    def test_400_point_difference(self):
        """Test 400 point difference produces ~91% expected."""
        expected = calculate_expected_win_chance(1900, 1500)
        # 10^(400/400) = 10, so expected = 1/(1+0.1) ≈ 0.909
        assert expected == pytest.approx(0.909, abs=0.01)


class TestUpdateElo:
    """Tests for Elo rating updates."""

    def test_winner_gains_loser_loses(self):
        """Test winner gains rating, loser loses rating."""
        new_a, new_b = update_elo(1500, 1500, "A", k_factor=32)

        assert new_a > 1500
        assert new_b < 1500
        # Zero-sum: gains equal losses
        assert (new_a - 1500) == pytest.approx(-(new_b - 1500), abs=0.01)

    def test_upset_win_larger_change(self):
        """Test upset win produces larger rating change."""
        # Underdog wins
        new_a, _new_b = update_elo(1400, 1600, "A", k_factor=32)

        # Underdog should gain more than expected
        assert new_a - 1400 > 16  # More than half of K

    def test_expected_win_smaller_change(self):
        """Test expected win produces smaller rating change."""
        # Favorite wins
        new_a, _new_b = update_elo(1600, 1400, "A", k_factor=32)

        # Favorite gains less than expected
        assert new_a - 1600 < 16

    def test_confidence_weighting(self):
        """Test confidence affects K-factor."""
        # High confidence
        high_a, _ = update_elo(1500, 1500, "A", k_factor=32, confidence=1.0)

        # Low confidence
        low_a, _ = update_elo(1500, 1500, "A", k_factor=32, confidence=0.0)

        # High confidence should produce larger change
        assert high_a - 1500 > low_a - 1500

    def test_confidence_formula(self):
        """Test confidence weighting formula."""
        # At confidence=0.0, effective_K = K * 0.5
        # At confidence=1.0, effective_K = K * 1.0
        new_a_high, _ = update_elo(1500, 1500, "A", k_factor=32, confidence=1.0)
        new_a_low, _ = update_elo(1500, 1500, "A", k_factor=32, confidence=0.0)

        # Ratio should be about 2:1
        high_gain = new_a_high - 1500
        low_gain = new_a_low - 1500
        assert high_gain / low_gain == pytest.approx(2.0, abs=0.01)


class TestEloRating:
    """Tests for EloRating dataclass."""

    def test_defaults(self):
        """Test default values."""
        rating = EloRating()
        assert rating.rating == 1500.0
        assert rating.matches == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.history == []

    def test_record_win(self):
        """Test recording a win."""
        rating = EloRating()
        rating.record_match(1516.0, won=True)

        assert rating.rating == 1516.0
        assert rating.matches == 1
        assert rating.wins == 1
        assert rating.losses == 0
        assert 1500.0 in rating.history

    def test_record_loss(self):
        """Test recording a loss."""
        rating = EloRating()
        rating.record_match(1484.0, won=False)

        assert rating.rating == 1484.0
        assert rating.matches == 1
        assert rating.wins == 0
        assert rating.losses == 1

    def test_history_tracking(self):
        """Test rating history is tracked."""
        rating = EloRating()
        rating.record_match(1520, won=True)
        rating.record_match(1540, won=True)
        rating.record_match(1530, won=False)

        assert len(rating.history) == 3
        assert rating.history == [1500.0, 1520, 1540]


class TestEloSystem:
    """Tests for EloSystem ranking."""

    def test_initialize(self):
        """Test initialization of candidates."""
        system = EloSystem(initial_rating=1500.0, k_factor=32.0)
        system.initialize(["a", "b", "c"])

        assert system.get_rating("a") == 1500.0
        assert system.get_rating("b") == 1500.0
        assert system.get_rating("c") == 1500.0

    def test_update_winner_gains(self):
        """Test that winner gains rating after match."""
        system = EloSystem()
        system.initialize(["a", "b"])

        system.update("a", "b", confidence=1.0)

        assert system.get_rating("a") > 1500.0
        assert system.get_rating("b") < 1500.0

    def test_update_zero_sum(self):
        """Test that rating changes are zero-sum."""
        system = EloSystem()
        system.initialize(["a", "b"])

        system.update("a", "b", confidence=1.0)

        gain_a = system.get_rating("a") - 1500.0
        loss_b = 1500.0 - system.get_rating("b")

        assert gain_a == pytest.approx(loss_b, abs=0.01)

    def test_stats_tracking(self):
        """Test match statistics are tracked."""
        system = EloSystem()
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
        system = EloSystem()
        system.initialize(["a", "b", "c"])

        # a beats b, b beats c
        system.update("a", "b", confidence=1.0)
        system.update("b", "c", confidence=1.0)

        leaderboard = system.get_leaderboard()

        # Should be sorted: a > b > c
        assert leaderboard[0][0] == "a"
        assert leaderboard[1][0] == "b"
        assert leaderboard[2][0] == "c"

    def test_confidence_affects_k_factor(self):
        """Test that confidence affects rating change magnitude."""
        system_high = EloSystem()
        system_high.initialize(["a", "b"])
        system_high.update("a", "b", confidence=1.0)
        high_change = system_high.get_rating("a") - 1500.0

        system_low = EloSystem()
        system_low.initialize(["a", "b"])
        system_low.update("a", "b", confidence=0.0)
        low_change = system_low.get_rating("a") - 1500.0

        # High confidence should produce 2x the change of low confidence
        assert high_change / low_change == pytest.approx(2.0, abs=0.01)

    def test_get_rating_object(self):
        """Test accessing full EloRating object."""
        system = EloSystem()
        system.initialize(["a"])

        system.update(
            "a", "a", confidence=1.0
        )  # Self-match won't happen but tests object access

        rating_obj = system.get_rating_object("a")
        assert isinstance(rating_obj, EloRating)
