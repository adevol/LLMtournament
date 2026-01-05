"""Swiss-style pairing for LLM Tournament."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A candidate in the tournament.

    Attributes:
        id: Unique identifier (essay_id).
        rating: Current rating (Elo or specialized).
        played_against: Set of opponent IDs already played.
        writer_slug: Original writer model slug.
        critic_slug: Critic slug if v1 essay, None if v0.
    """

    id: str
    rating: float = 1500.0
    played_against: set[str] = field(default_factory=set)
    writer_slug: str = ""
    critic_slug: str | None = None
    byes: int = 0


def swiss_pairing(
    candidates: list[Candidate],
    seed: int | None = None,
    bucket_width_ratio: float = 0.05,
) -> tuple[list[tuple[Candidate, Candidate]], Candidate | None]:
    """Generate Swiss-style pairings for one round.

    Swiss pairing matches players of similar skill while avoiding rematches.
    This implementation uses rating buckets to balance fairness with randomness:

    1. Candidates are sorted by rating (descending)
    2. They are grouped into "buckets" of similar ratings
    3. Candidates within each bucket are shuffled randomly
    4. Adjacent candidates in the shuffled list are paired

    Buckets are contiguous rating bands. For example, with ratings [1600, 1580,
    1550, 1400, 1380] and bucket_width=50, you get two buckets:
    - Bucket 1: [1600, 1580, 1550] (all within 50 of 1600)
    - Bucket 2: [1400, 1380] (all within 50 of 1400)

    This ensures similarly-rated candidates compete while adding randomness
    to prevent predictable brackets.

    When there's an odd number of candidates, one receives a "bye" (sits out).
    The bye recipient is returned so the caller can optionally award them a
    virtual win to keep ratings fair.

    Args:
        candidates: List of candidates to pair.
        seed: Random seed for reproducible shuffling.
        bucket_width_ratio: Bucket width as fraction of rating spread (default 5%).
            With a 200-point spread, buckets are 10 points wide.

    Returns:
        Tuple of (pairs, bye_recipient). bye_recipient is None if even count.
    """
    min_pair_size = 2
    if len(candidates) < min_pair_size:
        return [], None

    rng = random.Random(seed)  # noqa: S311

    pairing_pool = list(candidates)
    bye_recipient: Candidate | None = None
    if len(pairing_pool) % 2 == 1:
        bye_recipient = _assign_bye(pairing_pool)

    sorted_candidates = sorted(pairing_pool, key=lambda c: c.rating, reverse=True)

    ratings = [c.rating for c in sorted_candidates]
    spread = max(ratings) - min(ratings)
    bucket_width = max(spread * bucket_width_ratio, 1.0)

    buckets = _group_into_buckets(sorted_candidates, bucket_width)
    for bucket in buckets:
        rng.shuffle(bucket)

    shuffled = [c for bucket in buckets for c in bucket]
    pairs, unpaired = _create_pairs_from_shuffled(shuffled, allow_rematches=False)

    if len(unpaired) > 1:
        fallback_pairs, _ = _create_pairs_from_shuffled(unpaired, allow_rematches=True)
        pairs.extend(fallback_pairs)

    return pairs, bye_recipient


def _group_into_buckets(
    sorted_candidates: list[Candidate], bucket_width: float
) -> list[list[Candidate]]:
    """Group sorted candidates into contiguous rating buckets.

    Iterates through candidates (already sorted by rating descending) and groups
    them into buckets. A new bucket starts when a candidate's rating differs from
    the bucket's anchor rating by more than bucket_width.

    Args:
        sorted_candidates: Candidates sorted by rating (descending).
        bucket_width: Maximum rating difference within a bucket.

    Returns:
        List of buckets, each containing candidates with similar ratings.
    """
    buckets: list[list[Candidate]] = []
    current_bucket: list[Candidate] = []
    anchor_rating: float | None = None

    for candidate in sorted_candidates:
        if anchor_rating is None or abs(candidate.rating - anchor_rating) <= bucket_width:
            current_bucket.append(candidate)
            if anchor_rating is None:
                anchor_rating = candidate.rating
        else:
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = [candidate]
            anchor_rating = candidate.rating

    if current_bucket:
        buckets.append(current_bucket)

    return buckets


def _assign_bye(candidates: list[Candidate]) -> Candidate:
    """Remove one candidate to receive a bye (lowest rated with fewest byes).

    Args:
        candidates: Mutable list of candidates to select from.

    Returns:
        The candidate who received the bye.
    """
    candidates.sort(key=lambda c: c.rating)

    min_byes = min(c.byes for c in candidates)
    for i, c in enumerate(candidates):
        if c.byes == min_byes:
            c.byes += 1
            return candidates.pop(i)

    return candidates.pop(0)


def _create_pairs_from_shuffled(
    shuffled: list[Candidate], allow_rematches: bool = False
) -> tuple[list[tuple[Candidate, Candidate]], list[Candidate]]:
    """Pair adjacent candidates.

    Returns:
        Tuple of (pairs, unpaired_candidates)
    """
    pairs: list[tuple[Candidate, Candidate]] = []
    used: set[str] = set()

    for i, candidate_a in enumerate(shuffled):
        if candidate_a.id in used:
            continue

        for candidate_b in shuffled[i + 1 :]:
            if candidate_b.id in used:
                continue
            if not allow_rematches and candidate_b.id in candidate_a.played_against:
                continue

            pairs.append((candidate_a, candidate_b))
            used.add(candidate_a.id)
            used.add(candidate_b.id)
            break

    unpaired = [c for c in shuffled if c.id not in used]
    return pairs, unpaired


def create_candidates_v0(writer_slugs: list[str], initial_rating: float) -> list[Candidate]:
    """Create candidates from v0 essays."""
    return [
        Candidate(id=slug, rating=initial_rating, writer_slug=slug, critic_slug=None)
        for slug in writer_slugs
    ]


def create_candidates_v1(
    writer_slugs: list[str], critic_slugs: list[str], initial_rating: float
) -> list[Candidate]:
    """Create candidates from v1 essays."""
    candidates = []
    for writer in writer_slugs:
        for critic in critic_slugs:
            essay_id = f"{writer}__{critic}"
            candidates.append(
                Candidate(
                    id=essay_id,
                    rating=initial_rating,
                    writer_slug=writer,
                    critic_slug=critic,
                )
            )
    return candidates
