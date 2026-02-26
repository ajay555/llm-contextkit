"""Conversation history layer for ContextKit.

Manages conversation history with pluggable management strategies.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_contextkit.history.strategies import HistoryStrategy, Message, get_strategy
from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class HistoryLayer(BaseLayer):
    """Conversation history layer with pluggable management strategy.

    Manages conversation history using different strategies:
    - sliding_window: Keep last N turns
    - sliding_window_with_summary: Summarize older turns, keep recent
    - selective: Include only messages relevant to current query

    Args:
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
        strategy: History management strategy name. Default: "sliding_window".
        strategy_config: Dict of strategy-specific config options.
        name: Layer name. Default: "history".
        priority: Truncation priority. Default: 5.

    Strategy configs:
        sliding_window:
            {"max_turns": 10}

        sliding_window_with_summary:
            {"max_recent_turns": 5, "summarizer": callable_or_none}

        selective:
            {"query": str, "max_turns": 10, "relevance_threshold": 0.5}

    Examples:
        >>> layer = HistoryLayer(
        ...     messages=[
        ...         {"role": "user", "content": "Hello"},
        ...         {"role": "assistant", "content": "Hi there!"},
        ...     ],
        ...     strategy="sliding_window",
        ...     strategy_config={"max_turns": 10},
        ... )
        >>> content = layer.build(token_counter)
    """

    def __init__(
        self,
        messages: list[Message],
        strategy: str = "sliding_window",
        strategy_config: dict[str, Any] | None = None,
        name: str = "history",
        priority: int = 5,
    ) -> None:
        self._messages = [msg.copy() for msg in messages]  # Deep copy
        self._strategy_name = strategy
        self._strategy_config = strategy_config or {}
        self._strategy: HistoryStrategy = get_strategy(strategy, strategy_config)
        self._processed_messages: list[Message] = []
        self._strategy_metadata: dict[str, Any] = {}

        super().__init__(name=name, content=messages, priority=priority)

    @property
    def messages(self) -> list[Message]:
        """Return the original messages."""
        return [msg.copy() for msg in self._messages]

    @property
    def processed_messages(self) -> list[Message]:
        """Return the processed messages after build/truncate."""
        return [msg.copy() for msg in self._processed_messages]

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return self._strategy_name

    def build(self, token_counter: TokenCounter) -> str:
        """Build the history layer content.

        Applies the configured strategy to manage history.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted conversation history.
        """
        # Apply strategy with a large token budget initially
        # (actual truncation happens in truncate() if needed)
        max_tokens = 1_000_000  # Effectively unlimited for initial build
        self._processed_messages, self._strategy_metadata = self._strategy.apply(
            self._messages, max_tokens, token_counter
        )

        self._built_content = self._format_messages(self._processed_messages)
        self._token_count = token_counter.count(self._built_content)

        return self._built_content

    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        Re-applies the strategy with the token constraint.

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content.
        """
        original_count = len(self._processed_messages)

        # Re-apply strategy with token constraint
        self._processed_messages, self._strategy_metadata = self._strategy.apply(
            self._messages, max_tokens, token_counter
        )

        self._built_content = self._format_messages(self._processed_messages)
        self._token_count = token_counter.count(self._built_content)

        if len(self._processed_messages) < original_count:
            self._truncated = True
            logger.info(
                f"HistoryLayer '{self._name}': Truncated from {original_count} "
                f"to {len(self._processed_messages)} messages to fit within "
                f"{max_tokens} token budget"
            )

        return self._built_content

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages as conversation history.

        Args:
            messages: List of message dicts.

        Returns:
            Formatted conversation string.
        """
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"{role}: {content}")

        return "\n\n".join(lines)

    def get_messages_for_api(self) -> list[Message]:
        """Get processed messages in API format.

        Returns messages suitable for direct use with OpenAI/Anthropic APIs.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return self.processed_messages

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        base_info = super().inspect()
        base_info.update(
            {
                "strategy": self._strategy_name,
                "total_messages": len(self._messages),
                "processed_messages": len(self._processed_messages),
                "strategy_metadata": self._strategy_metadata,
            }
        )
        return base_info
