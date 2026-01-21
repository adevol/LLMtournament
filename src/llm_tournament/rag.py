"""Simple RAG system for document retrieval.

Uses markitdown for ingestion and sentence-transformers + numpy for retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from markitdown import MarkItDown
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@runtime_checkable
class Retriever(Protocol):
    """Protocol for document retrieval systems.

    Implement this protocol to provide custom retrieval logic.
    The `RAGSystem` class is an example implementation.
    """

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query.

        Args:
            query: The search query.
            top_k: Number of chunks to return.

        Returns:
            Retrieved context as a string.
        """
        ...


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            break

    return chunks


class RAGSystem:
    """Simple vector-based retrieval system.

    Loads documents using markitdown, chunks them, embeds with sentence-transformers,
    and retrieves relevant chunks via cosine similarity.
    """

    def __init__(
        self,
        source_path: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize RAG system.

        Args:
            source_path: Path to directory containing source documents.
            embedding_model: Name of the sentence-transformers model.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self.source_path = Path(source_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedder = SentenceTransformer(embedding_model)
        self._chunks: list[str] = []
        self._embeddings: NDArray[np.float32] | None = None
        self._is_indexed = False

    def load_and_index(self) -> None:
        """Load documents from source_path and build the index."""
        if not self.source_path.exists():
            msg = f"Source path does not exist: {self.source_path}"
            raise FileNotFoundError(msg)

        md = MarkItDown()
        all_text_parts: list[str] = []

        supported_extensions = {".txt", ".md", ".docx", ".doc", ".pdf"}
        for file_path in self.source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = md.convert(str(file_path))
                if result.text_content:
                    all_text_parts.append(result.text_content)

        full_text = "\n\n".join(all_text_parts)
        self._chunks = chunk_text(full_text, self.chunk_size, self.chunk_overlap)

        if self._chunks:
            self._embeddings = self._embedder.encode(self._chunks, convert_to_numpy=True)

        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The query string.
            top_k: Number of top results to return.

        Returns:
            Concatenated relevant chunks as a single string.
        """
        if not self._is_indexed:
            self.load_and_index()

        if not self._chunks or self._embeddings is None:
            return ""

        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_chunks = [self._chunks[i] for i in top_indices]

        return "\n\n---\n\n".join(relevant_chunks)
