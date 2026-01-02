"""Report generation services for LLM Tournament."""

from __future__ import annotations

from collections import defaultdict

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


def generate_writer_aggregate(
    candidates: list[Candidate],
    ranking_system: RankingSystem,
) -> str:
    """Generate writer aggregate report content.

    Args:
        candidates: List of candidates.
        ranking_system: Ranking system with current ratings.

    Returns:
        Markdown report content.
    """
    writer_ratings: dict[str, list[float]] = defaultdict(list)
    for c in candidates:
        writer_ratings[c.writer_slug].append(ranking_system.get_rating(c.id))

    lines = [
        "# Writer Aggregate Rankings",
        "",
        "| Writer | Mean Rating | Variants |",
        "| --- | --- | --- |",
    ]
    for writer, ratings in sorted(
        writer_ratings.items(), key=lambda x: -sum(x[1]) / len(x[1])
    ):
        mean_rating = sum(ratings) / len(ratings)
        lines.append(f"| {writer[:30]} | {mean_rating:.1f} | {len(ratings)} |")

    return "\n".join(lines)


def generate_critic_metrics(
    candidates: list[Candidate],
    ranking_system: RankingSystem,
) -> str:
    """Generate critic metrics report content.

    Args:
        candidates: List of candidates.
        ranking_system: Ranking system with current ratings.

    Returns:
        Markdown report content.
    """
    critic_ratings: dict[str, list[float]] = defaultdict(list)
    for c in candidates:
        if c.critic_slug:
            critic_ratings[c.critic_slug].append(ranking_system.get_rating(c.id))

    lines = [
        "# Critic Metrics",
        "",
        "Mean rating of essays revised using each critic's feedback.",
        "",
        "| Critic | Mean Rating | Essays |",
        "| --- | --- | --- |",
    ]
    for critic, ratings in sorted(
        critic_ratings.items(), key=lambda x: -sum(x[1]) / len(x[1])
    ):
        mean_rating = sum(ratings) / len(ratings)
        lines.append(f"| {critic[:30]} | {mean_rating:.1f} | {len(ratings)} |")

    return "\n".join(lines)
