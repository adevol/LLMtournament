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
MULTI_JUDGE = True  # Use multiple judges per match for higher reliability

# 2026 frontier models
WRITERS = [
    "minimax/minimax-m2.1",
    "z-ai/glm-4.7",
    "moonshotai/kimi-k2-thinking",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

# Use same models for critics and judges
CRITICS = WRITERS
JUDGES = WRITERS


async def main() -> None:
    """Run tournament with frontier models."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    # Threshold of 1.0 forces audit (2nd judge) on every match
    audit_threshold = 1.0 if MULTI_JUDGE else 0.7

    config = TournamentConfig(
        writers=WRITERS,
        critics=CRITICS,
        judges=JUDGES,
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
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,  # v0 only for speed
        seed=2026,
        token_caps={
            "writer_tokens": 1200,
            "critic_tokens": 200,
            "revision_tokens": 1300,
            "judge_tokens": 800,  # Increased to prevent truncation
        },
        ranking={
            "rounds": 5,
            "algorithm": "trueskill",
            "audit_confidence_threshold": audit_threshold,
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
