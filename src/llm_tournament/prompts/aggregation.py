"""Prompts for cross-topic aggregation analysis."""


def model_profile_system_prompt() -> str:
    """Return system prompt for model profiling."""
    return (
        "You are an expert evaluator of LLM performance. Your goal is to analyze "
        "how a specific model performed across multiple debate topics, identifying "
        "its core strengths, weaknesses, and behavioral patterns.\n"
        "\n"
        "Focus on:\n"
        "1. Argumentation quality (logic, evidence usage)\n"
        "2. Rhetorical style (tone, persuasion vs. analysis)\n"
        "3. Adaptability (did it handle different topics well?)\n"
        "4. Specific blind spots or tendencies\n"
        "\n"
        "Be concise, objective, and evidence-based."
    )


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

    return (
        f"Analyze the performance of model '{model_id}' across the following tournament topics:\n"
        f"{results_text}\n"
        "\n"
        "Synthesize a profile for this model with the following sections:\n"
        "## Executive Summary\n"
        "One paragraph overview of the model's overall capability level.\n\n"
        "## Key Strengths\n"
        "Bulleted list of what this model does consistently well.\n\n"
        "## Weaknesses & Limitations\n"
        "Bulleted list of where this model struggles or exhibits bias.\n\n"
        "## Strategic Advice\n"
        "What prompts or tasks is this model best suited for?"
    )


def cross_topic_insights_system_prompt() -> str:
    """Return system prompt for cross-topic insights."""
    return (
        "You are a meta-analyst for an LLM tournament. Your goal is to look at the "
        "aggregate results across all models and topics to identify high-level trends, "
        "anomalies, and insights about the current state of these frontier models.\n"
        "\n"
        "Focus on comparing models against each other and identifying what separates "
        "the winners from the losers."
    )


def cross_topic_insights_user_prompt(rankings: str, model_profiles: str) -> str:
    """Return user prompt for cross-topic insights."""
    return (
        "Based on the following aggregate rankings and individual model profiles, "
        "provide a cross-topic analysis of the tournament.\n\n"
        "### Aggregate Rankings\n"
        f"{rankings}\n\n"
        "### Model Profiles Summaries\n"
        f"{model_profiles}\n\n"
        "Please provide a report with:\n"
        "1. **The Hierarchy**: Clear tiers of model capability observed.\n"
        "2. **The Meta_Game**: What strategies were most successful across topics?\n"
        "3. **Topic Variance**: Which models were specialists vs generalists?\n"
        "4. **Unexpected Findings**: Any surprising upsets or behaviors."
    )
