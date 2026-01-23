#!/usr/bin/env python
"""Run a RAG-enhanced tournament simulation with multiple models.

Demonstrates how to use RAG to retrieve relevant context from source documents
and inject it into the prompts for LLM evaluation.

This example compares:
- Base models (no RAG context)
- RAG-enhanced models (with retrieved context)
- Custom RAG with different retrieval settings
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_tournament.core.config import TournamentConfig, WriterConfig
from llm_tournament.pipeline import TournamentPipeline
from llm_tournament.rag import RAGSystem, Retriever
from llm_tournament.services.llm import CostTracker, PricingService, create_client
from llm_tournament.services.storage import TournamentStore

load_dotenv()

# Path to RAG source documents
RAG_SOURCES_DIR = Path(__file__).parent / "rag_sources" / "economy_2025"


class LargeChunkRetriever:
    """Custom retriever with larger chunks for more context.

    Demonstrates how to create a custom Retriever implementation.
    """

    def __init__(self, source_path: Path) -> None:
        self._rag = RAGSystem(
            source_path,
            chunk_size=1000,  # Larger chunks
            chunk_overlap=100,
        )
        self._rag.load_and_index()

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve with more chunks for broader context."""
        return self._rag.retrieve(query, top_k=top_k)


# Verify protocol compliance
assert isinstance(LargeChunkRetriever(RAG_SOURCES_DIR), Retriever)


async def main() -> None:
    """Run RAG-enhanced tournament with multiple models."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    print("Building RAG systems...")

    # Standard RAG system
    rag_standard = RAGSystem(RAG_SOURCES_DIR)
    rag_standard.load_and_index()

    # Define models to compare
    # Mix of base models, RAG-enhanced, and custom RAG
    writers_config = [
        # Base models (no RAG)
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-haiku",
        # RAG-enhanced versions (standard retrieval)
        WriterConfig(
            model_id="openai/gpt-4o-mini",
            name="gpt-4o-mini-rag",
            use_rag=True,
        ),
        WriterConfig(
            model_id="google/gemini-2.0-flash-001",
            name="gemini-2.0-flash-rag",
            use_rag=True,
        ),
        WriterConfig(
            model_id="anthropic/claude-3.5-haiku",
            name="claude-3.5-haiku-rag",
            use_rag=True,
        ),
    ]

    # Judge panel - use a mix for diverse perspectives
    judges = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
    ]

    config = TournamentConfig(
        writers=writers_config,
        critics=["openai/gpt-4o-mini"],
        judges=judges,
        retriever=rag_standard,  # Primary retriever for RAG writers
        topics=[
            {
                "title": "2025-2026 Economic Analysis",
                "prompts": {
                    "Essay": (
                        "Write an analysis of the key economic forces shaping 2025-2026. "
                        "Focus on the interplay between fiscal policy and market valuations. "
                        "Discuss specific trends and provide concrete examples."
                    ),
                },
                "rag_queries": {
                    "Essay": (
                        "What are the key economic drivers, market trends, "
                        "and fiscal policy changes in 2025-2026?"
                    ),
                },
            },
            {
                "title": "AI Investment Landscape",
                "prompts": {
                    "Essay": (
                        "Analyze the AI investment landscape in 2025-2026. "
                        "Discuss major players, funding trends, and potential risks."
                    ),
                },
                "rag_queries": {
                    "Essay": (
                        "What are the AI investment trends, major companies, "
                        "and market dynamics in 2025-2026?"
                    ),
                },
            },
        ],
        output_dir=str(Path("./runs")),
        simple_mode=True,
        seed=2026,
        writer_tokens=2500,
        judge_tokens=1200,
        ranking={
            "rounds": 4,  # More rounds for stable rankings with more participants
            "algorithm": "elo",
            "judging_method": "audit",
        },
        api_key=api_key,
    )

    store = TournamentStore(config)
    print(f"Initialized tournament run: {store.run_id}")
    print(f"Comparing {len(writers_config)} writers across {len(config.topics)} topics")

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
        pipeline = TournamentPipeline(config, client, store, max_concurrency=2)
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
