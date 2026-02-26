"""Base layer abstract class for ContextKit.

All context layers inherit from BaseLayer and implement the build() and
truncate() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_contextkit.tokenizers.counting import TokenCounter


class BaseLayer(ABC):
    """Abstract base class for all context layers.

    Every layer must implement:
    - build(): Assemble the layer content as a formatted string
    - truncate(max_tokens): Reduce content to fit within token budget

    Layers are immutable after creation. To modify content, create a new layer.

    Args:
        name: Unique identifier for this layer.
        content: The raw content for this layer (type varies by layer type).
        priority: Priority for truncation ordering. Higher priority layers
            are truncated last. Default: 0.

    Attributes:
        name: The layer's unique identifier.
        content: The raw content (immutable after init).
        priority: Truncation priority.
    """

    def __init__(self, name: str, content: Any, priority: int = 0) -> None:
        self._name = name
        self._content = content
        self._priority = priority
        self._built_content: str | None = None
        self._token_count: int = 0
        self._truncated: bool = False

    @property
    def name(self) -> str:
        """Return the layer name."""
        return self._name

    @property
    def content(self) -> Any:
        """Return the raw layer content."""
        return self._content

    @property
    def priority(self) -> int:
        """Return the layer priority."""
        return self._priority

    @property
    def built_content(self) -> str | None:
        """Return the built content, or None if not yet built."""
        return self._built_content

    @property
    def truncated(self) -> bool:
        """Return whether this layer was truncated."""
        return self._truncated

    @abstractmethod
    def build(self, token_counter: TokenCounter) -> str:
        """Build the formatted layer content.

        This method assembles the layer's raw content into a formatted string
        suitable for inclusion in the context. It also updates the internal
        token count.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted layer content as a string.
        """
        pass

    @abstractmethod
    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        This method reduces the layer content to fit within the specified
        token budget. It sets the _truncated flag if any content was removed.

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content as a string.
        """
        pass

    def get_token_count(self) -> int:
        """Return token count of built content.

        Returns:
            The number of tokens in the built content, or 0 if not built.
        """
        return self._token_count

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        content_preview = None
        if self._built_content:
            content_preview = (
                self._built_content[:200] + "..."
                if len(self._built_content) > 200
                else self._built_content
            )

        return {
            "name": self._name,
            "priority": self._priority,
            "token_count": self._token_count,
            "content_preview": content_preview,
            "truncated": self._truncated,
        }
