"""Tests for Swiss-style pairing."""

from llm_tournament.services.match.pairing import (
    Candidate,
    create_candidates_v0,
    create_candidates_v1,
    swiss_pairing,
)


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_candidate_defaults(self):
        """Test candidate has correct defaults."""
        c = Candidate(id="test")
        assert c.rating == 1500.0
        assert c.played_against == set()
        assert c.writer_slug == ""
        assert c.critic_slug is None


class TestSwissPairing:
    """Tests for Swiss pairing algorithm."""

    def test_pairs_adjacent_by_elo(self):
        """Test candidates are paired adjacently by Elo."""
        candidates = [
            Candidate(id="a", rating=1600),
            Candidate(id="b", rating=1500),
            Candidate(id="c", rating=1400),
            Candidate(id="d", rating=1300),
        ]

        pairs, _ = swiss_pairing(candidates, seed=42)

        assert len(pairs) == 2
        pair_ids = [{p[0].id, p[1].id} for p in pairs]
        assert {"a", "b"} in pair_ids or {"a", "c"} in pair_ids

    def test_avoids_repeat_matchups(self):
        """Test already-played pairs are not repeated."""
        candidates = [
            Candidate(id="a", rating=1600, played_against={"b"}),
            Candidate(id="b", rating=1550, played_against={"a"}),
            Candidate(id="c", rating=1500),
            Candidate(id="d", rating=1450),
        ]

        pairs, _ = swiss_pairing(candidates, seed=42)

        for c1, c2 in pairs:
            assert c2.id not in c1.played_against

    def test_deterministic_with_seed(self):
        """Test pairing is deterministic with same seed."""
        candidates = [
            Candidate(id="a", rating=1500),
            Candidate(id="b", rating=1500),
            Candidate(id="c", rating=1500),
            Candidate(id="d", rating=1500),
        ]

        pairs1, _ = swiss_pairing(candidates, seed=42)
        pairs2, _ = swiss_pairing(candidates, seed=42)

        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2, strict=True):
            assert {p1[0].id, p1[1].id} == {p2[0].id, p2[1].id}

    def test_odd_number_leaves_one_unpaired(self):
        """Test odd number of candidates leaves one unpaired."""
        candidates = [
            Candidate(id="a", rating=1600),
            Candidate(id="b", rating=1500),
            Candidate(id="c", rating=1400),
        ]

        pairs, bye = swiss_pairing(candidates, seed=42)

        assert len(pairs) == 1
        assert bye is not None
        paired_ids = {pairs[0][0].id, pairs[0][1].id}
        assert bye.id not in paired_ids

    def test_empty_candidates_returns_empty(self):
        """Test empty input returns empty list."""
        pairs, bye = swiss_pairing([], seed=42)
        assert pairs == []
        assert bye is None

    def test_single_candidate_returns_empty(self):
        """Test single candidate returns empty list."""
        pairs, bye = swiss_pairing([Candidate(id="a")], seed=42)
        assert pairs == []
        assert bye is None

    def test_bye_allocation_logic(self):
        """Test correct candidate is selected for bye."""
        # a: 1600 (0 byes)
        # b: 1500 (0 byes)
        # c: 1400 (0 byes) -> Should get bye (lowest rating)
        candidates = [
            Candidate(id="a", rating=1600),
            Candidate(id="b", rating=1500),
            Candidate(id="c", rating=1400),
        ]

        pairs, bye = swiss_pairing(candidates, seed=42)

        assert len(pairs) == 1
        assert bye is not None
        assert bye.id == "c"
        pair_ids = {pairs[0][0].id, pairs[0][1].id}
        assert "a" in pair_ids
        assert "b" in pair_ids
        assert bye.byes == 1

        c_candidate = next(cand for cand in candidates if cand.id == "c")
        c_candidate.byes = 1
        pairs_round_2, bye2 = swiss_pairing(candidates, seed=43)

        assert len(pairs_round_2) == 1
        assert bye2 is not None
        assert bye2.id == "b"
        pair2_ids = {pairs_round_2[0][0].id, pairs_round_2[0][1].id}
        assert "a" in pair2_ids
        assert "c" in pair2_ids
        assert bye2.byes == 1

    def test_fallback_rematches(self):
        """Test fallback to rematch when strict pairing fails."""
        # a played b, c.
        # b played a, c.
        # c played a, b.
        # d played None.

        candidates = [
            Candidate(id="a", rating=1600, played_against={"b", "c"}),
            Candidate(id="b", rating=1600, played_against={"a", "c"}),
            Candidate(id="c", rating=1600, played_against={"a", "b"}),
            Candidate(id="d", rating=1600, played_against=set()),
        ]

        pairs, _ = swiss_pairing(candidates, seed=42)

        assert len(pairs) == 2
        paired_ids = {p[0].id for p in pairs} | {p[1].id for p in pairs}
        assert len(paired_ids) == 4


class TestCreateCandidates:
    """Tests for candidate creation functions."""

    def test_create_candidates_v0(self):
        """Test v0 candidate creation."""
        writers = ["writer-a", "writer-b"]
        candidates = create_candidates_v0(writers, initial_rating=1400)

        assert len(candidates) == 2
        assert candidates[0].id == "writer-a"
        assert candidates[0].rating == 1400
        assert candidates[0].critic_slug is None

    def test_create_candidates_v1(self):
        """Test v1 candidate creation (cross product)."""
        writers = ["w1", "w2"]
        critics = ["c1", "c2", "c3"]
        candidates = create_candidates_v1(writers, critics, initial_rating=1500)

        assert len(candidates) == 6  # 2 * 3
        ids = {c.id for c in candidates}
        assert "w1__c1" in ids
        assert "w2__c3" in ids

    def test_v1_tracks_writer_and_critic(self):
        """Test v1 candidates track writer and critic slugs."""
        candidates = create_candidates_v1(["writer"], ["critic"], initial_rating=1500)

        assert len(candidates) == 1
        assert candidates[0].writer_slug == "writer"
        assert candidates[0].critic_slug == "critic"
