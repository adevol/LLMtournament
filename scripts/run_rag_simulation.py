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

from llm_tournament.core.config import TournamentConfig, WriterConfig
from llm_tournament.pipeline import TournamentPipeline
from llm_tournament.rag import RAGSystem
from llm_tournament.services.llm import CostTracker, PricingService, create_client
from llm_tournament.services.storage import TournamentStore

load_dotenv()

# Path to RAG source documents
RAG_SOURCES_DIR = Path(__file__).parent / "rag_sources" / "economy_2025"

JUDGES = ["openai/gpt-4o-mini"]


async def main() -> None:
    """Run RAG-enhanced tournament."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    print("Building RAG context...")

    # Initialize your retriever
    rag = RAGSystem(RAG_SOURCES_DIR)
    rag.load_and_index()

    # Compare: Base model vs RAG-enhanced model
    writers_config = [
        # 1. Base model (no context)
        "openai/gpt-4o-mini",
        # 2. RAG-enhanced model (with retrieved context from topic)
        WriterConfig(
            model_id="openai/gpt-4o-mini",
            name="gpt-4o-mini-rag",
            use_rag=True,
        ),
    ]

    config = TournamentConfig(
        writers=writers_config,
        critics=["openai/gpt-4o-mini"],
        judges=JUDGES,
        retriever=rag,
        topics=[
            {
                "title": "2025-2026 Economic Analysis",
                "prompts": {
                    "Essay": (
                        "Write an analysis of the key economic forces shaping 2025-2026. "
                        "Focus on the interplay between fiscal policy and market valuations."
                    ),
                },
                # RAG queries per prompt key - retrieval happens automatically
                "rag_queries": {
                    "Essay": "What are the key economic drivers and market trends in 2025-2026?",
                },
                "source_pack": None,
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,
        seed=2026,
        writer_tokens=2000,
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
