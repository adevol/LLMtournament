"""Report generation services for LLM Tournament."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Literal

from tabulate import tabulate

from llm_tournament.core.config import truncate_slug
from llm_tournament.models import Rating
from llm_tournament.ranking import RankingSystem
from llm_tournament.services.match import Candidate


def build_rating_objects(
    leaderboard: list,
    candidates: list[Candidate],
    algorithm: str,
) -> list[Rating]:
    """Convert leaderboard entries to Rating objects.

    Args:
        leaderboard: Raw leaderboard from ranking system.
        candidates: List of candidates.
        algorithm: Ranking algorithm name ("elo" or "trueskill").

    Returns:
        List of Rating objects.
    """
    is_trueskill = algorithm == "trueskill"
    candidate_map = {c.id: c for c in candidates}
    rating_objects = []

    for entry in leaderboard:
        if is_trueskill:
            cid, rating, wins, losses, mu, sigma = entry
        else:
            cid, rating, wins, losses = entry
            mu, sigma = None, None

        c = candidate_map[cid]
        rating_obj = Rating(
            candidate_id=cid,
            rating=rating,
            matches=wins + losses,
            wins=wins,
            losses=losses,
            writer_slug=c.writer_slug,
            critic_slug=c.critic_slug,
            mu=mu,
            sigma=sigma,
        )
        rating_objects.append(rating_obj)

    return rating_objects


def generate_aggregate_report(
    candidates: list[Candidate],
    ranking_system: RankingSystem,
    group_by: Literal["writer", "critic"],
    title: str,
    headers: tuple[str, str, str],
    max_slug_length: int,
    description: str | None = None,
) -> str:
    """Generate an aggregate report grouped by a candidate attribute.

    Args:
        candidates: List of candidates.
        ranking_system: Ranking system with current ratings.
        group_by: Attribute to group by ("writer" or "critic").
        title: Report title (markdown heading).
        headers: Column headers (name, rating, count).
        max_slug_length: Maximum slug length for report labels.
        description: Optional description line below title.

    Returns:
        Markdown report content.
    """
    attr = f"{group_by}_slug"
    # Group ratings by the specified attribute
    ratings_by_group: dict[str, list[float]] = defaultdict(list)
    for c in candidates:
        group_key = getattr(c, attr)
        if group_key:
            ratings_by_group[group_key].append(ranking_system.get_rating(c.id))

    # Sort by mean rating descending and build table rows
    def by_mean_desc(item: tuple[str, list[float]]) -> float:
        return -mean(item[1])

    rows = [
        (truncate_slug(group, max_slug_length), f"{mean(ratings):.1f}", len(ratings))
        for group, ratings in sorted(ratings_by_group.items(), key=by_mean_desc)
    ]

    lines = [f"# {title}", ""]
    if description:
        lines.extend([description, ""])
    lines.append(tabulate(rows, headers=headers, tablefmt="github"))

    return "\n".join(lines)
