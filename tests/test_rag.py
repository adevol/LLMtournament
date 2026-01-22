"""Tests for RAG module."""

import tempfile
from pathlib import Path

import pytest

from llm_tournament.rag import RAGSystem, Retriever, build_rag_writer, chunk_text


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


class TestBuildRagWriter:
    """Tests for build_rag_writer helper."""

    def test_creates_valid_writer_config(self, tmp_path: Path):
        """Returns a valid WriterConfig dict."""
        (tmp_path / "doc.txt").write_text("Test content about economics.")
        rag = RAGSystem(tmp_path)
        rag.load_and_index()

        result = build_rag_writer(
            model_id="openai/gpt-4o",
            retriever=rag,
            query="economics",
        )

        assert result["model_id"] == "openai/gpt-4o"
        assert result["name"] == "gpt-4o-rag"
        assert "system_prompt" in result
        assert (
            "economics" in result["system_prompt"].lower()
            or "content" in result["system_prompt"].lower()
        )

    def test_custom_name(self, tmp_path: Path):
        """Custom name is used correctly."""
        (tmp_path / "doc.txt").write_text("Test content.")
        rag = RAGSystem(tmp_path)

        result = build_rag_writer(
            model_id="openai/gpt-4o",
            retriever=rag,
            query="test",
            name="my-custom-rag",
        )

        assert result["name"] == "my-custom-rag"

    def test_custom_prompt_template(self, tmp_path: Path):
        """Custom prompt template is used."""
        (tmp_path / "doc.txt").write_text("Custom content here.")
        rag = RAGSystem(tmp_path)

        result = build_rag_writer(
            model_id="openai/gpt-4o",
            retriever=rag,
            query="test",
            prompt_template="CONTEXT: {context} END",
        )

        assert "CONTEXT:" in result["system_prompt"]
        assert "END" in result["system_prompt"]
