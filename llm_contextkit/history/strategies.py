"""History management strategies for ContextKit.

Provides different strategies for managing conversation history:
- sliding_window: Keep last N turns
- sliding_window_with_summary: Summarize older turns, keep recent
- selective: Include only messages relevant to current query
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, TypedDict

from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class Message(TypedDict):
    """A conversation message."""

    role: str  # "user" or "assistant"
    content: str


class HistoryStrategy(ABC):
    """Abstract base class for history management strategies."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the strategy with optional configuration."""
        self._config = config or {}

    @abstractmethod
    def apply(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Apply the strategy to manage history.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens for the history.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            A tuple of (processed_messages, metadata).
            metadata contains strategy-specific information about what was done.
        """
        pass


class SlidingWindowStrategy(HistoryStrategy):
    """Keep the last N turns of conversation.

    A turn is a user message followed by an assistant response.

    Config:
        max_turns: Maximum number of turns to keep. Default: 10.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._max_turns = config.get("max_turns", 10)

    def apply(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Apply sliding window strategy.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens for the history.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            Tuple of (filtered_messages, metadata).
        """
        if not messages:
            return [], {"turns_included": 0, "turns_dropped": 0, "strategy": "sliding_window"}

        # Count turns (each user-assistant pair is one turn)
        total_turns = self._count_turns(messages)

        # Calculate how many turns to keep
        turns_to_keep = min(self._max_turns, total_turns)

        # Get messages for the last N turns
        kept_messages = self._get_last_n_turns(messages, turns_to_keep)

        # Further truncate if needed to fit token budget
        kept_messages, tokens_truncated = self._truncate_to_tokens(
            kept_messages, max_tokens, token_counter
        )

        final_turns = self._count_turns(kept_messages)
        metadata = {
            "turns_included": final_turns,
            "turns_dropped": total_turns - final_turns,
            "strategy": "sliding_window",
            "max_turns_config": self._max_turns,
            "token_truncated": tokens_truncated,
        }

        return kept_messages, metadata

    def _count_turns(self, messages: list[Message]) -> int:
        """Count conversation turns."""
        # Count user messages as turns (each user message starts a turn)
        return sum(1 for m in messages if m["role"] == "user")

    def _get_last_n_turns(self, messages: list[Message], n: int) -> list[Message]:
        """Get messages from the last N turns."""
        if n <= 0:
            return []

        # Find indices of user messages (turn starts)
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        if len(user_indices) <= n:
            return messages.copy()

        # Get the index where we should start (nth turn from the end)
        start_idx = user_indices[-n]
        return messages[start_idx:]

    def _truncate_to_tokens(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        """Truncate messages to fit within token budget."""
        if not messages:
            return [], False

        # Try all messages first
        total_tokens = self._count_message_tokens(messages, token_counter)
        if total_tokens <= max_tokens:
            return messages, False

        # Need to truncate - remove oldest turns first
        truncated = True
        result = messages.copy()

        while result and self._count_message_tokens(result, token_counter) > max_tokens:
            # Find and remove the first complete turn
            if result and result[0]["role"] == "user":
                result.pop(0)
                # Also remove following assistant message if present
                if result and result[0]["role"] == "assistant":
                    result.pop(0)
            elif result:
                result.pop(0)

        return result, truncated

    def _count_message_tokens(
        self, messages: list[Message], token_counter: TokenCounter
    ) -> int:
        """Count total tokens in messages."""
        total = 0
        for msg in messages:
            # Count role label + content
            total += token_counter.count(f"{msg['role']}: {msg['content']}")
        return total


class SlidingWindowWithSummaryStrategy(HistoryStrategy):
    """Keep recent turns and summarize older turns.

    Config:
        max_recent_turns: Number of recent turns to keep in full. Default: 5.
        summarizer: Optional callable (messages) -> str for summarization.
            If None, uses basic extractive summary (first sentence of each message).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._max_recent_turns = config.get("max_recent_turns", 5)
        self._summarizer: Callable[[list[Message]], str] | None = config.get("summarizer")

    def apply(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Apply sliding window with summary strategy.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens for the history.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            Tuple of (processed_messages, metadata).
        """
        if not messages:
            return [], {
                "turns_included": 0,
                "turns_summarized": 0,
                "strategy": "sliding_window_with_summary",
            }

        # Split into older and recent messages
        older_messages, recent_messages = self._split_messages(messages)

        # Summarize older messages if any
        summary = ""
        if older_messages:
            summary = self._create_summary(older_messages)

        # Build result with summary (if any) + recent messages
        result_messages: list[Message] = []

        if summary:
            # Add summary as a system-like context message
            result_messages.append({
                "role": "assistant",
                "content": f"[Previous conversation summary: {summary}]",
            })

        result_messages.extend(recent_messages)

        # Truncate if needed
        result_messages = self._truncate_to_tokens(result_messages, max_tokens, token_counter)

        recent_turns = self._count_turns(recent_messages)
        summarized_turns = self._count_turns(older_messages)

        metadata = {
            "turns_included": recent_turns,
            "turns_summarized": summarized_turns,
            "strategy": "sliding_window_with_summary",
            "summary_included": bool(summary),
        }

        return result_messages, metadata

    def _count_turns(self, messages: list[Message]) -> int:
        """Count conversation turns."""
        return sum(1 for m in messages if m["role"] == "user")

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
        """Split messages into older and recent."""
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        if len(user_indices) <= self._max_recent_turns:
            return [], messages.copy()

        # Split at the nth turn from the end
        split_idx = user_indices[-self._max_recent_turns]
        return messages[:split_idx], messages[split_idx:]

    def _create_summary(self, messages: list[Message]) -> str:
        """Create a summary of messages."""
        if self._summarizer:
            return self._summarizer(messages)

        # Basic extractive summary: first sentence of each message
        summaries = []
        for msg in messages:
            content = msg["content"]
            # Get first sentence
            first_sentence = content.split(".")[0].strip()
            if first_sentence:
                role = msg["role"].capitalize()
                summaries.append(f"{role}: {first_sentence}.")

        return " ".join(summaries[:5])  # Limit to first 5

    def _truncate_to_tokens(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> list[Message]:
        """Truncate messages to fit within token budget."""
        result = messages.copy()

        while result:
            total = sum(
                token_counter.count(f"{m['role']}: {m['content']}") for m in result
            )
            if total <= max_tokens:
                break
            # Remove oldest message
            result.pop(0)

        return result


class SelectiveStrategy(HistoryStrategy):
    """Include only messages relevant to the current query.

    Config:
        query: The current query to match against.
        max_turns: Maximum turns to include. Default: 10.
        relevance_threshold: Minimum relevance score (0-1). Default: 0.5.
        similarity_fn: Optional callable (query, message) -> float for scoring.
            If None, uses basic keyword matching.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._query = config.get("query", "")
        self._max_turns = config.get("max_turns", 10)
        self._relevance_threshold = config.get("relevance_threshold", 0.5)
        self._similarity_fn: Callable[[str, str], float] | None = config.get(
            "similarity_fn"
        )

    def apply(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Apply selective strategy based on relevance.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens for the history.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            Tuple of (relevant_messages, metadata).
        """
        if not messages or not self._query:
            return messages[-self._max_turns * 2 :] if messages else [], {
                "turns_included": 0,
                "turns_dropped": 0,
                "strategy": "selective",
            }

        # Score each message for relevance
        scored_messages: list[tuple[float, int, Message]] = []
        for i, msg in enumerate(messages):
            score = self._compute_relevance(msg["content"])
            scored_messages.append((score, i, msg))

        # Filter by threshold and sort by original order
        relevant = [
            (score, idx, msg)
            for score, idx, msg in scored_messages
            if score >= self._relevance_threshold
        ]

        # Keep original order
        relevant.sort(key=lambda x: x[1])

        # Limit to max turns worth of messages
        result_messages: list[Message] = []
        turn_count = 0
        for _score, _idx, msg in relevant:
            if msg["role"] == "user":
                turn_count += 1
            if turn_count > self._max_turns:
                break
            result_messages.append(msg)

        # Truncate to token budget
        result_messages = self._truncate_to_tokens(
            result_messages, max_tokens, token_counter
        )

        total_turns = sum(1 for m in messages if m["role"] == "user")
        included_turns = sum(1 for m in result_messages if m["role"] == "user")

        metadata = {
            "turns_included": included_turns,
            "turns_dropped": total_turns - included_turns,
            "strategy": "selective",
            "query": self._query[:50] + "..." if len(self._query) > 50 else self._query,
        }

        return result_messages, metadata

    def _compute_relevance(self, text: str) -> float:
        """Compute relevance score for text against query."""
        if self._similarity_fn:
            return self._similarity_fn(self._query, text)

        # Basic keyword matching
        query_words = set(self._query.lower().split())
        text_words = set(text.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & text_words)
        return overlap / len(query_words)

    def _truncate_to_tokens(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> list[Message]:
        """Truncate messages to fit within token budget."""
        result = messages.copy()

        while result:
            total = sum(
                token_counter.count(f"{m['role']}: {m['content']}") for m in result
            )
            if total <= max_tokens:
                break
            result.pop(0)

        return result


def get_strategy(name: str, config: dict[str, Any] | None = None) -> HistoryStrategy:
    """Get a history strategy by name.

    Args:
        name: Strategy name: "sliding_window", "sliding_window_with_summary",
            or "selective".
        config: Strategy-specific configuration.

    Returns:
        A HistoryStrategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    strategies: dict[str, type[HistoryStrategy]] = {
        "sliding_window": SlidingWindowStrategy,
        "sliding_window_with_summary": SlidingWindowWithSummaryStrategy,
        "selective": SelectiveStrategy,
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown history strategy: {name}. "
            f"Available strategies: {list(strategies.keys())}"
        )

    strategy_class = strategies[name]
    return strategy_class(config)
