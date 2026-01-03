"""Prompts for cross-topic aggregation analysis."""

from llm_tournament.prompts import get_prompt_group


def _get_aggregation_prompts() -> dict[str, dict[str, str]]:
    """Safe getter for nested aggregation prompts."""
    # Ensure backward compatibility if prompts are missing or file reload is needed
    return get_prompt_group("aggregation")


_AGG_PROMPTS = _get_aggregation_prompts()


def model_profile_system_prompt() -> str:
    """Return system prompt for model profiling."""
    return _AGG_PROMPTS["model_profile"]["system"]


def model_profile_user_prompt(model_id: str, topic_results: list[dict]) -> str:
    """Return user prompt for model profiling.

    Args:
        model_id: The ID of the model being analyzed.
        topic_results: List of dicts, each containing:
            - topic: title of the topic
            - rank: ranking in that topic (e.g., "1/6")
            - score: numerical score/rating
            - summary: brief summary of performance/feedback for that topic
    """
    results_text = ""
    for res in topic_results:
        results_text += f"\n## Topic: {res['topic']}\n"
        results_text += f"Rank: {res['rank']}\n"
        results_text += f"Rating: {res['score']:.1f}\n"
        results_text += f"Performance Summary:\n{res['summary']}\n"

    template = _AGG_PROMPTS["model_profile"]["user_template"]
    return template.format(model_id=model_id, results_text=results_text)


def cross_topic_insights_system_prompt() -> str:
    """Return system prompt for cross-topic insights."""
    return _AGG_PROMPTS["cross_topic_insights"]["system"]


def cross_topic_insights_user_prompt(rankings: str, model_profiles: str) -> str:
    """Return user prompt for cross-topic insights."""
    template = _AGG_PROMPTS["cross_topic_insights"]["user_template"]
    return template.format(rankings=rankings, model_profiles=model_profiles)
