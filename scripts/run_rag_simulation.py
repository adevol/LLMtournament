#!/usr/bin/env python
"""Run a simple RAG-enhanced tournament simulation.

Demonstrates how to use RAG to retrieve relevant context from source documents
and inject it into the prompts for LLM evaluation.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_tournament.core.config import TournamentConfig
from llm_tournament.pipeline import TournamentPipeline
from llm_tournament.rag import RAGSystem
from llm_tournament.services.llm import CostTracker, PricingService, create_client
from llm_tournament.services.storage import TournamentStore

load_dotenv()

# Path to RAG source documents
RAG_SOURCES_DIR = Path(__file__).parent / "rag_sources" / "economy_2025"

# Models to test (using fewer for demonstration)
WRITERS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
]

CRITICS = WRITERS

JUDGES = [
    "openai/gpt-4o-mini",
]


def build_rag_context(query: str, top_k: int = 3) -> str:
    """Build RAG context from source documents.

    Args:
        query: The question or topic to retrieve context for.
        top_k: Number of chunks to retrieve.

    Returns:
        Retrieved context as a formatted string.
    """
    rag = RAGSystem(RAG_SOURCES_DIR)
    rag.load_and_index()
    return rag.retrieve(query, top_k=top_k)


async def main() -> None:
    """Run RAG-enhanced tournament."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    # Build RAG context for each topic
    print("Building RAG context from source documents...")

    economy_context = build_rag_context(
        "What are the key economic drivers and market trends in 2025-2026?"
    )
    dollar_context = build_rag_context(
        "How did the dollar perform against other currencies and gold?"
    )
    market_context = build_rag_context("How did US stocks compare to international markets?")

    print(f"Retrieved {len(economy_context)} chars of economy context")
    print(f"Retrieved {len(dollar_context)} chars of dollar context")
    print(f"Retrieved {len(market_context)} chars of market context")

    config = TournamentConfig(
        writers=WRITERS,
        critics=CRITICS,
        judges=JUDGES,
        topics=[
            {
                "title": "2025-2026 Economic Analysis",
                "prompts": {
                    "Essay": (
                        "Based on the source material provided, write an analysis of the "
                        "key economic forces shaping 2025-2026. Focus on the interplay "
                        "between fiscal policy, monetary policy, and market valuations. "
                        "What are the main risks and opportunities for investors?"
                    ),
                },
                "source_pack": economy_context,
            },
            {
                "title": "Dollar and Currency Dynamics",
                "prompts": {
                    "Essay": (
                        "Using the provided source material, analyze the dollar's "
                        "performance in 2025 and its implications for global investors. "
                        "Discuss how currency movements affected real returns across "
                        "different asset classes."
                    ),
                },
                "source_pack": dollar_context,
            },
            {
                "title": "Global Market Rotation",
                "prompts": {
                    "Essay": (
                        "Based on the source data, explain why non-US markets "
                        "outperformed US stocks in 2025. What does this mean for "
                        "portfolio construction and global asset allocation going forward?"
                    ),
                },
                "source_pack": market_context,
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,  # v0 only for speed
        seed=2026,
        writer_tokens=2000,
        critic_tokens=800,
        revision_tokens=2000,
        judge_tokens=1000,
        ranking={
            "rounds": 3,
            "algorithm": "elo",
            "judging_method": "audit",
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
        pipeline = TournamentPipeline(config, client, store, max_concurrency=4)
        await pipeline.run()

        # Log costs
        total_cost = await cost_tracker.get_total_cost()
        print(f"\nTotal Cost: ${total_cost:.4f}")

        costs_path = store.base_dir / "final_analysis" / "costs.json"
        costs_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown = await cost_tracker.get_cost_breakdown()
        model_costs = await cost_tracker.get_model_costs()
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
