#!/usr/bin/env python
"""Run a college application essay tournament with fictional applicant context.

This script uses supporting documents in scripts/college_assets and asks
frontier models to generate and critique two graduate application essays.
"""

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

ASSETS_DIR = Path(__file__).resolve().parent / "college_assets"

WRITERS = [
    "minimax/minimax-m2.1",
    "z-ai/glm-4.7",
    "moonshotai/kimi-k2-thinking",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

CRITICS = WRITERS

PRIMARY_JUDGES = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

SUB_JUDGES = [
    "moonshotai/kimi-k2-thinking",
    "z-ai/glm-4.7",
]

JUDGES = PRIMARY_JUDGES + SUB_JUDGES


def load_asset(filename: str) -> str:
    path = ASSETS_DIR / filename
    return path.read_text(encoding="utf-8").strip()


def build_source_pack() -> str:
    sections = [
        ("Student Profile", load_asset("student_profile.md")),
        ("CV (Markdown)", load_asset("cv.md")),
        ("Career Aspirations", load_asset("aspirations.md")),
        ("Research Interests", load_asset("interests.md")),
    ]
    parts = []
    for title, content in sections:
        parts.append(f"{title}\n{'-' * len(title)}\n{content}")
    return "\n\n".join(parts)


ESSAY_CONSTRAINTS = (
    "Write 400 to 500 words. Use specific details from the "
    "source material. Avoid cliches and generic claims. Do not use headings "
    "or bullet points."
)


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    source_pack = build_source_pack()

    config = TournamentConfig(
        writers=WRITERS,
        critics=CRITICS,
        judges=JUDGES,
        topics=[
            {
                "title": "Academic and Professional Motivation",
                "prompts": {
                    "Essay": (
                        "Describe your academic/professional motivation for applying "
                        "to this program. Draw upon your past and present work and "
                        "academic experiences as well as aspirations and goals for the "
                        "future. " + ESSAY_CONSTRAINTS
                    ),
                },
                "source_pack": source_pack,
            },
            {
                "title": "Why MIT and Georgia Tech",
                "prompts": {
                    "Essay": (
                        "Tell us why you want to pursue your graduate education at MIT. "
                        "Consider including questions or issues that inspire you, "
                        "experiences that have shaped your professional interests, and "
                        "why you think that MIT is well-suited to help you. "
                        "Include information that may assist faculty in evaluating "
                        "your preparation and aptitude for graduate education at "
                        "MIT. " + ESSAY_CONSTRAINTS
                    ),
                },
                "source_pack": source_pack,
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=False,
        seed=2026,
        writer_tokens=2200,
        critic_tokens=900,
        revision_tokens=2200,
        judge_tokens=1200,
        ranking={
            "rounds": 7,
            "algorithm": "trueskill",
            "judging_method": "parallel_majority",
            "audit_confidence_threshold": 0.7,
            "primary_judges": PRIMARY_JUDGES,
            "sub_judges": SUB_JUDGES,
        },
        api_key=api_key,
    )

    store = TournamentStore(config)
    print("College Application Essay Tournament")
    print("=" * 40)
    print(f"Run ID: {store.run_id}")
    print(f"Assets: {ASSETS_DIR}")
    print(f"Models: {len(WRITERS)} writers, {len(CRITICS)} critics")

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

        total_cost = await cost_tracker.get_total_cost()
        print(f"\nTotal Cost: ${total_cost:.4f}")
        breakdown = await cost_tracker.get_cost_breakdown()
        model_costs = await cost_tracker.get_model_costs()
        costs_path = store.base_dir / "final_analysis" / "costs.json"
        costs_path.parent.mkdir(parents=True, exist_ok=True)
        costs_path.write_text(
            json.dumps(
                {
                    "total_usd": total_cost,
                    "by_role": breakdown,
                    "by_model": model_costs,
                    "models": WRITERS,
                    "topics": [t.title for t in config.topics],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"\nTournament complete. Results in: {store.base_dir}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
