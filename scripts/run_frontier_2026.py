#!/usr/bin/env python
"""Run LLM tournament with 2026 frontier models."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_tournament.core.config import TournamentConfig
from llm_tournament.pipeline import run_tournament
from llm_tournament.services.llm import create_client

load_dotenv()

# Configuration flags
PARALLEL_MAJORITY = True  # Use 3-judge parallel voting (vs sequential audit mode)

# 2026 frontier models
WRITERS = [
    "minimax/minimax-m2.1",
    "z-ai/glm-4.7",
    "moonshotai/kimi-k2-thinking",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

# Use same models for critics
CRITICS = WRITERS

# Primary judges (3 used in parallel for initial voting)
PRIMARY_JUDGES = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

# Sub-judges (2 added if confidence is low)
SUB_JUDGES = [
    "moonshotai/kimi-k2-thinking",
    "z-ai/glm-4.7",
]

# All judges (for fallback/rotation in audit mode)
JUDGES = PRIMARY_JUDGES + SUB_JUDGES


async def main() -> None:
    """Run tournament with frontier models."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    judging_method = "parallel_majority" if PARALLEL_MAJORITY else "audit"

    config = TournamentConfig(
        writers=WRITERS,
        critics=CRITICS,
        judges=JUDGES,
        # Topics designed to test nuanced reasoning:
        # - Arguing against popular positions while steel-manning opponents
        # - Understanding second-order economic effects
        # - Avoiding both sycophantic agreement and contrarian overreach
        topics=[
            {
                "title": "The Future of Work",
                "prompts": {
                    "Essay": (
                        "Write a comprehensive essay exploring how AI and automation "
                        "will transform employment over the next decade. Consider both "
                        "opportunities and challenges, and propose strategies for "
                        "workforce adaptation."
                    ),
                },
                "source_pack": (
                    "Key concepts: automation displacement, skills gap, universal "
                    "basic income, human-AI collaboration, labor market polarization."
                ),
            },
            {
                # Tests: Arguing against a popular ESG narrative while being fair
                # to opposing views. Reveals if AI defaults to safe consensus.
                "title": "ESG Investing Underperformance",
                "prompts": {
                    "Essay": (
                        "Explain why ESG investing may systematically underperform "
                        "traditional investing, while acknowledging the strongest "
                        "arguments in favor of ESG. Be rigorous about economic "
                        "mechanisms rather than ideological."
                    ),
                },
                "source_pack": (
                    "Key concepts: portfolio constraints, factor tilts, exclusion costs, "
                    "greenwashing, fiduciary duty, stranded assets, risk-adjusted returns."
                ),
            },
            {
                # Tests: Understanding counterintuitive long-run effects in housing
                # markets. Reveals if AI can reason about dynamic equilibria.
                "title": "Short-Term Rental Bans",
                "prompts": {
                    "Essay": (
                        "Argue why banning short-term rentals (like Airbnb) may hurt "
                        "renters in the long run, even if rents fall initially. "
                        "Address the strongest counterarguments about housing supply "
                        "and neighborhood character."
                    ),
                },
                "source_pack": (
                    "Key concepts: housing supply elasticity, investment incentives, "
                    "property rights, neighborhood effects, hotel industry competition, "
                    "dynamic vs static analysis, unintended consequences."
                ),
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,  # v0 only for speed
        seed=2026,
        token_caps={
            "writer_tokens": 3500,
            "critic_tokens": 1500,
            "revision_tokens": 3500,
            "judge_tokens": 2000,
        },
        ranking={
            "rounds": 5,  # optional, auto-calculated as ceil(log2(N)) + 1 for stable rankings
            "algorithm": "trueskill",
            "judging_method": judging_method,
            "audit_confidence_threshold": 0.7,
            "primary_judges": PRIMARY_JUDGES,
            "sub_judges": SUB_JUDGES,  # optional, only needed for parallel_majority judging
        },
        api_key=api_key,
    )

    client = create_client(
        api_key=api_key,
        cache_path=Path("./runs/.cache/llm_cache.duckdb"),
        use_cache=True,
    )

    try:
        store = await run_tournament(config, client, max_concurrency=3)
        print(f"Tournament complete! Results in: {store.base_dir}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
