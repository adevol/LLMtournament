#!/usr/bin/env python
"""Run LLM tournament with 2026 frontier models."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_tournament.core.config import TournamentConfig
from llm_tournament.pipeline import TournamentPipeline
from llm_tournament.services.llm import CostTracker, PricingService, create_client
from llm_tournament.services.storage import TournamentStore

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
                "title": "The Lantern of Hollowmere",
                "prompts": {
                    "Story": (
                        "Write a short story (900-1200 words) set in a DnD fantasy world. "
                        "Center on a traveling party arriving at a fogbound fishing town "
                        "and the discovery of a cursed lantern. Include dialogue, a moral "
                        "choice, and a closing hook for the next session."
                    ),
                },
                "source_pack": (
                    "Key concepts: coastal village, ancient bargain, wary locals, "
                    "supernatural fog, price of salvation."
                ),
            },
            {
                "title": "The Obsidian Vault Heist",
                "prompts": {
                    "Scene": (
                        "Write a tense infiltration scene (600-900 words) in a DnD dungeon. "
                        "Show the rogue, cleric, wizard, and fighter coordinating in whispers. "
                        "Include a trap with a clear trigger, a risky skill check moment, and "
                        "a twist that forces a fast choice."
                    ),
                },
                "source_pack": (
                    "Key concepts: pressure plates, arcane wards, time limit, silent signals, "
                    "unexpected guardian."
                ),
            },
            {
                "title": "Campfire Oath",
                "prompts": {
                    "Monologue": (
                        "Write a first-person monologue (500-700 words) from a paladin at a "
                        "campfire after a failed rescue. Reveal backstory, a renewed vow, and "
                        "a hint of inner corruption without naming it outright."
                    ),
                },
                "source_pack": (
                    "Key concepts: oathbound duty, guilt, temptation, mercy versus justice, "
                    "party trust."
                ),
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,  # v0 only for speed
        seed=2026,
        writer_tokens=3500,
        critic_tokens=1500,
        revision_tokens=3500,
        judge_tokens=2000,
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

    store = TournamentStore(config)
    print(f"Initialized tournament run: {store.run_id}")

    # Cost Tracking Setup
    pricing = PricingService(api_key)
    await pricing.refresh()
    cost_tracker = CostTracker(pricing, store._engine)

    client = create_client(
        api_key=api_key,
        cache_path=Path("./runs/.cache/llm_cache.duckdb"),
        use_cache=True,
        cost_tracker=cost_tracker,
    )

    try:
        pipeline = TournamentPipeline(config, client, store, max_concurrency=6)
        await pipeline.run()

        # Log costs
        total_cost = await cost_tracker.get_total_cost()
        print(f"\nTotal Cost: ${total_cost:.4f}")
        breakdown = await cost_tracker.get_cost_breakdown()
        model_costs = await cost_tracker.get_model_costs()
        print("Cost Breakdown:", breakdown)
        print("Cost by Model:", model_costs)
        costs_path = store.base_dir / "final_analysis" / "costs.json"
        costs_path.parent.mkdir(parents=True, exist_ok=True)
        costs_path.write_text(
            json.dumps(
                {
                    "total_usd": total_cost,
                    "by_role": breakdown,
                    "by_model": model_costs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Tournament complete! Results in: {store.base_dir}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
