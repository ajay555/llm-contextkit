"""Token counting utilities for ContextKit.

Supports tiktoken for accurate OpenAI token counting, with a fallback
approximate counter when tiktoken is not installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

from llm_contextkit.exceptions import TokenizerError

logger = logging.getLogger("contextkit")

# Mapping from friendly names to tiktoken encoding names
TOKENIZER_MAPPING: dict[str, str] = {
    "cl100k": "cl100k_base",  # GPT-4, GPT-3.5-turbo
    "o200k": "o200k_base",  # GPT-4o
}

# Approximate tokens per word ratio (conservative estimate)
APPROX_TOKENS_PER_WORD: float = 1.3


class TokenCounter(ABC):
    """Abstract base class for token counters."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this tokenizer."""
        pass


class TiktokenCounter(TokenCounter):
    """Token counter using tiktoken library.

    Args:
        encoding_name: The tiktoken encoding name (e.g., "cl100k_base", "o200k_base").
    """

    def __init__(self, encoding_name: str) -> None:
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
            self._name = encoding_name
        except ImportError as e:
            raise TokenizerError(
                "tiktoken is not installed. Install with: pip install contextkit[tiktoken]",
                tokenizer_name=encoding_name,
            ) from e
        except Exception as e:
            raise TokenizerError(
                f"Failed to load tiktoken encoding '{encoding_name}': {e}",
                tokenizer_name=encoding_name,
            ) from e

    def count(self, text: str) -> int:
        """Count tokens using tiktoken.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        return len(self._encoding.encode(text))

    @property
    def name(self) -> str:
        """Return the encoding name."""
        return self._name


class ApproximateCounter(TokenCounter):
    """Approximate token counter using word count estimation.

    Uses a simple heuristic: tokens ~= words * 1.3

    This is a fallback when tiktoken is not installed.
    """

    def __init__(self) -> None:
        self._name = "approximate"

    def count(self, text: str) -> int:
        """Count approximate tokens based on word count.

        Args:
            text: The text to count tokens in.

        Returns:
            The approximate number of tokens in the text.
        """
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * APPROX_TOKENS_PER_WORD)

    @property
    def name(self) -> str:
        """Return 'approximate'."""
        return self._name


class CallableCounter(TokenCounter):
    """Token counter using a user-provided callable.

    Args:
        counter_fn: A callable that takes a string and returns an int.
        name: Name to identify this counter.
    """

    def __init__(self, counter_fn: Callable[[str], int], name: str = "custom") -> None:
        self._counter_fn = counter_fn
        self._name = name

    def count(self, text: str) -> int:
        """Count tokens using the provided callable.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens as returned by the callable.
        """
        return self._counter_fn(text)

    @property
    def name(self) -> str:
        """Return the counter name."""
        return self._name


def get_token_counter(tokenizer: str | Callable[[str], int] = "cl100k") -> TokenCounter:
    """Get a token counter by name or callable.

    Args:
        tokenizer: Either a string identifying the tokenizer ("cl100k", "o200k",
            "approximate") or a callable that takes a string and returns an int.
            Defaults to "cl100k".

    Returns:
        A TokenCounter instance.

    Raises:
        TokenizerError: If the tokenizer cannot be loaded.

    Examples:
        >>> counter = get_token_counter("cl100k")
        >>> counter.count("Hello world")
        2

        >>> counter = get_token_counter("approximate")
        >>> counter.count("Hello world")
        3

        >>> counter = get_token_counter(lambda x: len(x.split()))
        >>> counter.count("Hello world")
        2
    """
    if callable(tokenizer):
        return CallableCounter(tokenizer)

    if tokenizer == "approximate":
        logger.warning(
            "Using approximate token counting. For accurate counts, "
            "install tiktoken: pip install contextkit[tiktoken]"
        )
        return ApproximateCounter()

    # Try to use tiktoken
    if tokenizer in TOKENIZER_MAPPING:
        encoding_name = TOKENIZER_MAPPING[tokenizer]
    else:
        encoding_name = tokenizer

    try:
        return TiktokenCounter(encoding_name)
    except TokenizerError:
        # Fall back to approximate counting with a warning
        logger.warning(
            "tiktoken not installed, using approximate token counting. "
            "Install with: pip install contextkit[tiktoken]"
        )
        return ApproximateCounter()
