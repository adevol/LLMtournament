"""Tests for RAG module."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from llm_tournament.core.config import TopicConfig, TournamentConfig, WriterConfig
from llm_tournament.prompts import writer_system_prompt
from llm_tournament.rag import RAGSystem, Retriever, build_rag_context, chunk_text
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.llm.client import LLMMessages, LLMResponse
from llm_tournament.services.submission import SubmissionService


class TestChunkText:
    """Tests for chunk_text function."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_text("") == []

    def test_short_text(self):
        """Text shorter than chunk_size returns single chunk."""
        result = chunk_text("Hello world", chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_chunks_with_overlap(self):
        """Overlapping chunks share characters."""
        text = "A" * 100
        result = chunk_text(text, chunk_size=50, overlap=10)
        # With 100 chars, chunk_size=50, overlap=10:
        # Chunk 1: 0-50, Chunk 2: 40-90, Chunk 3: 80-100
        assert len(result) >= 2

    def test_whitespace_only_chunks_excluded(self):
        """Chunks with only whitespace are excluded."""
        text = "Hello" + " " * 100 + "World"
        result = chunk_text(text, chunk_size=10, overlap=2)
        # Should not include empty chunks
        assert all(chunk.strip() for chunk in result)


class TestRetrieverProtocol:
    """Tests for Retriever protocol."""

    def test_ragsystem_implements_retriever(self):
        """RAGSystem satisfies the Retriever protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content for retrieval")

            rag = RAGSystem(tmpdir)
            assert isinstance(rag, Retriever)


class TestRAGSystem:
    """Tests for RAGSystem class."""

    @pytest.fixture
    def temp_source_dir(self, tmp_path: Path) -> Path:
        """Create temporary directory with test documents."""
        (tmp_path / "doc1.txt").write_text("The quick brown fox jumps over the lazy dog.")
        (tmp_path / "doc2.md").write_text("# AI and Machine Learning\n\nArtificial intelligence.")
        return tmp_path

    def test_load_and_index(self, temp_source_dir: Path):
        """Documents are loaded and chunked."""
        rag = RAGSystem(temp_source_dir, chunk_size=50, chunk_overlap=10)
        rag.load_and_index()
        assert len(rag._chunks) > 0
        assert rag._is_indexed

    def test_retrieve_returns_relevant_content(self, temp_source_dir: Path):
        """Retrieve returns content matching the query."""
        rag = RAGSystem(temp_source_dir, chunk_size=50, chunk_overlap=10)
        rag.load_and_index()

        result = rag.retrieve("fox", top_k=1)
        assert "fox" in result.lower() or "brown" in result.lower()

    def test_retrieve_auto_indexes(self, temp_source_dir: Path):
        """Retrieve triggers indexing if not done."""
        rag = RAGSystem(temp_source_dir)
        assert not rag._is_indexed

        rag.retrieve("test", top_k=1)
        assert rag._is_indexed

    def test_missing_source_path_raises(self, tmp_path: Path):
        """FileNotFoundError raised for missing source path."""
        rag = RAGSystem(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            rag.load_and_index()


class DummyRetriever:
    """Simple retriever for testing RAG workflows."""

    def retrieve(self, query: str, _top_k: int = 3) -> str:
        return f"retrieved:{query}"


class CapturingLLMClient(LLMClient):
    """LLM client stub that captures messages for assertions."""

    def __init__(self) -> None:
        self.messages: list[LLMMessages] = []

    @property
    def total_cost(self) -> float:
        return 0.0

    async def complete(
        self,
        _model: str,
        messages: LLMMessages,
        _max_tokens: int,
        _temperature: float,
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(content="ok", prompt_tokens=0, completion_tokens=0, total_tokens=0)


class DummyStore:
    """Store stub that records essays without touching disk."""

    def __init__(self) -> None:
        self.essays = self
        self.saved: list[tuple[str, str, str, str]] = []

    async def save_essay(
        self, topic_slug: str, writer_slug: str, content: str, version: str
    ) -> None:
        self.saved.append((topic_slug, writer_slug, content, version))


class TestRAGWorkflow:
    """Tests for the preferred WriterConfig(use_rag=True) workflow."""

    async def test_use_rag_injects_context(self):
        """RAG context is prepended when use_rag=True and queries exist."""
        topic = TopicConfig(
            title="Test Topic",
            prompts={"fiction": "Write a short story."},
            rag_queries={"fiction": "economics"},
        )
        writer = WriterConfig(model_id="openai/gpt-4o", use_rag=True)
        config = TournamentConfig(
            writers=[writer],
            critics=["openai/gpt-4o"],
            judges=["openai/gpt-4o"],
            topics=[topic],
        )
        config.retriever = DummyRetriever()

        client = CapturingLLMClient()
        store = DummyStore()
        service = SubmissionService(config, client, store, asyncio.Semaphore(1))

        await service.run_generation_batch(topic, config.get_writer_specs())

        assert client.messages, "Expected at least one LLM call"
        system_prompt = client.messages[0][0]["content"]
        rag_context = build_rag_context(config.retriever, "economics")
        base_prompt = writer_system_prompt()
        assert system_prompt.startswith(rag_context)
        assert system_prompt.endswith(base_prompt)

    async def test_use_rag_without_query_skips_context(self):
        """No context is injected when the topic lacks matching rag_queries."""
        topic = TopicConfig(
            title="Test Topic",
            prompts={"fiction": "Write a short story."},
            rag_queries={"journalism": "economics"},
        )
        writer = WriterConfig(model_id="openai/gpt-4o", use_rag=True)
        config = TournamentConfig(
            writers=[writer],
            critics=["openai/gpt-4o"],
            judges=["openai/gpt-4o"],
            topics=[topic],
        )
        config.retriever = DummyRetriever()

        client = CapturingLLMClient()
        store = DummyStore()
        service = SubmissionService(config, client, store, asyncio.Semaphore(1))

        await service.run_generation_batch(topic, config.get_writer_specs())

        assert client.messages, "Expected at least one LLM call"
        system_prompt = client.messages[0][0]["content"]
        assert system_prompt == writer_system_prompt()
