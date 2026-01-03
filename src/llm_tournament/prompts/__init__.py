"""Prompt templates for LLM Tournament.

Loads prompts from 'prompts.yaml' in the parent directory.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml

from llm_tournament.core.config import TopicConfig

logger = structlog.get_logger()

# Load prompts from single source of truth in package
# Using parent.parent because this is now in prompts/__init__.py
PROMPTS_PATH = Path(__file__).parent.parent / "prompts.yaml"


def _load_prompts() -> dict[str, str]:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_PATH}")

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Invalid prompts file: {PROMPTS_PATH} (must be dict)")
        return data


# Load on import
_PROMPTS = _load_prompts()


def writer_system_prompt() -> str:
    """System prompt for essay writers."""
    return _PROMPTS["writer_system"]


def writer_user_prompt(topic: TopicConfig) -> dict[str, str]:
    """Generate user prompts for essay writing (split by genre).

    Returns:
        Dictionary mapping genre 'fiction'|'journalism'|'scientific' to prompt string.
    """
    source_info = ""
    if topic.source_pack:
        template = _PROMPTS["source_material_block"]
        source_info = template.format(source_pack=topic.source_pack)

    prompts = {}

    template = _PROMPTS["writer_section_user"]

    for section, instruction in topic.prompts.items():
        # Inject source info if available
        # (We could optimize this to only include source_pack in Scientific
        # or just always include it if present)
        # The template has {source_info}, so we pass it.

        # Determine if source info should be shown.
        # Previously only for generic/scientific. Now generic.
        current_source_info = source_info if topic.source_pack else ""

        content = template.format(
            title=topic.title, instruction=instruction, source_info=current_source_info
        )
        prompts[section] = content

    return prompts


def critic_system_prompt() -> str:
    """System prompt for critics."""
    return _PROMPTS["critic_system"]


def critic_user_prompt(essay: str) -> str:
    """Generate user prompt for critique."""
    return _PROMPTS["critic_user"].format(essay=essay)


def revision_system_prompt() -> str:
    """System prompt for revision."""
    return _PROMPTS["revision_system"]


def revision_user_prompt(original_essay: str, feedback: str) -> str:
    """Generate user prompt for revision."""
    return _PROMPTS["revision_user"].format(
        original_essay=original_essay, feedback=feedback
    )


def judge_system_prompt() -> str:
    """System prompt for judges."""
    return _PROMPTS["judge_system"]


def judge_user_prompt(essay_a: str, essay_b: str) -> str:
    """Generate user prompt for judging."""
    return _PROMPTS["judge_user"].format(essay_a=essay_a, essay_b=essay_b)


def judge_strict_retry_prompt(essay_a: str, essay_b: str) -> str:
    """Generate stricter retry prompt when JSON parsing fails."""
    # This prompt contains double braces {{ }} for JSON in the template
    return _PROMPTS["judge_strict_retry"].format(essay_a=essay_a, essay_b=essay_b)


def analysis_system_prompt() -> str:
    """System prompt for post-ranking analysis."""
    return _PROMPTS["analysis_system"]


def analysis_user_prompt(entity_name: str, match_summaries: list[str]) -> str:
    """Generate user prompt for analysis."""
    summaries_text = "\n\n".join(
        f"Match {i + 1}:\n{s}" for i, s in enumerate(match_summaries)
    )
    return _PROMPTS["analysis_user"].format(
        entity_name=entity_name, summaries_text=summaries_text
    )
