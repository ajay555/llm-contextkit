"""Retrieved documents layer for ContextKit.

Handles RAG (Retrieval-Augmented Generation) chunks with metadata.
"""

import logging
from typing import Any, TypedDict

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class RetrievedChunk(TypedDict, total=False):
    """A retrieved document chunk."""

    text: str  # Required
    source: str  # Optional
    score: float  # Optional relevance score
    metadata: dict[str, Any]  # Optional additional metadata


class RetrievedLayer(BaseLayer):
    """Retrieved documents/chunks layer for RAG results.

    Formats retrieved chunks with metadata (source, relevance score) for
    inclusion in the context.

    Truncation behavior:
    - Chunks are removed from the end (lowest relevance first, assuming
      chunks are pre-sorted by relevance)
    - Chunks are never truncated partially - either included in full or dropped

    Args:
        chunks: List of dicts with at minimum {"text": str}.
            Optional fields: "source", "score", "metadata".
        max_chunks: Maximum chunks to include. Default: 5.
        include_metadata: Whether to include source/score info. Default: True.
        name: Layer name. Default: "retrieved_docs".
        priority: Truncation priority. Default: 6.

    Examples:
        >>> layer = RetrievedLayer(
        ...     chunks=[
        ...         {"text": "Policy document content...", "source": "policy.pdf", "score": 0.92},
        ...         {"text": "FAQ content...", "source": "faq.md", "score": 0.85},
        ...     ],
        ...     max_chunks=5,
        ... )
        >>> content = layer.build(token_counter)
    """

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        max_chunks: int = 5,
        include_metadata: bool = True,
        name: str = "retrieved_docs",
        priority: int = 6,
    ) -> None:
        self._chunks = [chunk.copy() for chunk in chunks]  # Deep copy
        self._max_chunks = max_chunks
        self._include_metadata = include_metadata
        self._included_chunks: int = 0

        super().__init__(name=name, content=chunks, priority=priority)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        """Return the original chunks."""
        return [chunk.copy() for chunk in self._chunks]

    @property
    def max_chunks(self) -> int:
        """Return the max chunks setting."""
        return self._max_chunks

    @property
    def include_metadata(self) -> bool:
        """Return whether metadata is included."""
        return self._include_metadata

    def build(self, token_counter: TokenCounter) -> str:
        """Build the retrieved docs layer content.

        Formats chunks with clear delimiters, source attribution, and
        relevance scores.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted retrieved documents content.
        """
        # Limit to max_chunks
        chunks_to_include = self._chunks[: self._max_chunks]

        self._built_content = self._format_chunks(chunks_to_include)
        self._token_count = token_counter.count(self._built_content)
        self._included_chunks = len(chunks_to_include)

        return self._built_content

    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        Removes chunks from the end (lowest relevance) until within budget.
        Never truncates a chunk partially.

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content.
        """
        # Start with max_chunks limit
        chunks_to_include = self._chunks[: self._max_chunks]

        # Remove chunks from end until we fit
        while chunks_to_include:
            content = self._format_chunks(chunks_to_include)
            if token_counter.count(content) <= max_tokens:
                break
            chunks_to_include.pop()  # Remove last chunk (lowest relevance)

        original_count = min(len(self._chunks), self._max_chunks)
        if len(chunks_to_include) < original_count:
            self._truncated = True
            dropped = original_count - len(chunks_to_include)
            logger.info(
                f"RetrievedLayer '{self._name}': Dropped {dropped} chunks "
                f"to fit within {max_tokens} token budget"
            )

        self._built_content = self._format_chunks(chunks_to_include)
        self._token_count = token_counter.count(self._built_content)
        self._included_chunks = len(chunks_to_include)

        return self._built_content

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks with metadata.

        Args:
            chunks: List of chunk dicts.

        Returns:
            Formatted chunks string.
        """
        if not chunks:
            return ""

        formatted_chunks = []
        for _i, chunk in enumerate(chunks, 1):
            chunk_parts = []

            # Add metadata header if enabled
            if self._include_metadata:
                meta_parts = []
                if "source" in chunk:
                    meta_parts.append(f"Source: {chunk['source']}")
                if "score" in chunk:
                    meta_parts.append(f"Relevance: {chunk['score']:.2f}")

                if meta_parts:
                    chunk_parts.append(f"[{' | '.join(meta_parts)}]")

            # Add the chunk text
            chunk_parts.append(chunk["text"])

            formatted_chunks.append("\n".join(chunk_parts))

        return "\n\n---\n\n".join(formatted_chunks)

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        base_info = super().inspect()
        base_info.update(
            {
                "total_chunks": len(self._chunks),
                "max_chunks_config": self._max_chunks,
                "included_chunks": self._included_chunks,
                "dropped_chunks": min(len(self._chunks), self._max_chunks)
                - self._included_chunks,
                "include_metadata": self._include_metadata,
            }
        )
        return base_info
