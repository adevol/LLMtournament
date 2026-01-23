"""LLM Tournament Evaluator.

Compare OpenRouter models via essay writing, critique, revision,
and pairwise ranking with Elo or TrueSkill.
"""

from llm_tournament.rag import RAGSystem, Retriever, build_rag_context

__version__ = "0.2.0"
__all__ = [
    "RAGSystem",
    "Retriever",
    "__version__",
    "build_rag_context",
]
