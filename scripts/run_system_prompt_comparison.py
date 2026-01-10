#!/usr/bin/env python
"""Run system prompt comparison tournament with Kimi K2.

This script demonstrates comparing the same model with different system prompts
to evaluate how prompt engineering affects output quality.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_tournament.core.config import TournamentConfig, WriterConfig
from llm_tournament.pipeline import TournamentPipeline
from llm_tournament.services.llm import CostTracker, PricingService, create_client
from llm_tournament.services.storage import TournamentStore

load_dotenv()

# The model to compare across different system prompts
MODEL_ID = "moonshotai/kimi-k2-thinking"

# Default system prompt (from prompts.yaml, exposed here for visibility)
# This can be overridden tournament-wide via TournamentConfig.writer_system_prompt,
# or per-writer via WriterConfig.system_prompt (which takes priority)
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert writer producing high-quality content. "
    "Do not include citations unless source material is provided. "
    "Focus on clarity, originality, and coherence."
)

# System prompts to compare
SYSTEM_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "analytical": (
        "You are an analytical writer who prioritizes logical structure and evidence-based "
        "reasoning. Present arguments systematically, acknowledge counterpoints, and draw "
        "conclusions only from the data provided. Avoid emotional language."
    ),
    "creative": (
        "You are a creative writer with a vivid, engaging style. Use metaphors, analogies, "
        "and storytelling techniques to make complex topics accessible and memorable. "
        "Balance creativity with accuracy."
    ),
    "concise": (
        "You are a concise, direct writer. Every sentence must add value. Eliminate filler "
        "words, redundant phrases, and unnecessary qualifiers. Prioritize clarity and "
        "brevity without sacrificing substance."
    ),
}

# Build writer configs from system prompts
WRITERS = [
    WriterConfig(
        model_id=MODEL_ID,
        system_prompt=prompt,
        name=f"kimi-{style}",
    )
    for style, prompt in SYSTEM_PROMPTS.items()
]

# Use diverse judges to reduce bias
JUDGES = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

# Use the same model as critic (unbiased since it's the same for all)
CRITICS = [MODEL_ID]


async def main() -> None:
    """Run system prompt comparison tournament."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    print("System Prompt Comparison Tournament")
    print("=" * 50)
    print(f"Model: {MODEL_ID}")
    print(f"Variants: {len(WRITERS)}")
    for writer in WRITERS:
        print(f"  - {writer.name}")
    print()

    config = TournamentConfig(
        writers=WRITERS,
        critics=CRITICS,
        judges=JUDGES,
        topics=[
            {
                "title": "Climate Adaptation Strategies",
                "prompts": {
                    "Essay": (
                        "Write an essay exploring effective strategies for communities "
                        "to adapt to climate change impacts. Consider infrastructure, "
                        "policy, and behavioral changes. Be specific about tradeoffs."
                    ),
                },
                "source_pack": (
                    "Key concepts: resilience, managed retreat, green infrastructure, "
                    "heat islands, flood mitigation, drought planning, equity."
                ),
            },
            {
                "title": "Remote Work Future",
                "prompts": {
                    "Essay": (
                        "Analyze the long-term implications of widespread remote work "
                        "on urban planning, social structures, and economic geography. "
                        "Consider both benefits and hidden costs."
                    ),
                },
                "source_pack": (
                    "Key concepts: urban exodus, commercial real estate, social capital, "
                    "productivity measurement, work-life boundaries, digital divide."
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
            "rounds": 4,
            "algorithm": "trueskill",
            "judging_method": "parallel_majority",
            "audit_confidence_threshold": 0.7,
            "primary_judges": JUDGES,
        },
        api_key=api_key,
    )

    store = TournamentStore(config)
    print(f"Run ID: {store.run_id}")

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
        pipeline = TournamentPipeline(config, client, store, max_concurrency=4)
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
                    "system_prompts": SYSTEM_PROMPTS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"\nTournament complete! Results in: {store.base_dir}")
        print("\nThis run compared these system prompt styles:")
        for style, prompt in SYSTEM_PROMPTS.items():
            print(f"  {style}: {prompt[:60]}...")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
