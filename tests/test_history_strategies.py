"""Tests for contextkit.history.strategies module."""

import pytest

from llm_contextkit.history.strategies import (
    SelectiveStrategy,
    SlidingWindowStrategy,
    SlidingWindowWithSummaryStrategy,
    get_strategy,
)
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


@pytest.fixture
def sample_messages():
    """Provide sample conversation messages."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great, thanks for asking!"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "It's sunny and warm today."},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
        {"role": "user", "content": "How do I read a file?"},
        {"role": "assistant", "content": "Use open() and read() functions."},
    ]


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    def test_default_config(self, token_counter, sample_messages):
        """Test with default configuration."""
        strategy = SlidingWindowStrategy()
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Default max_turns is 10, we have 4 turns
        assert len(result) == 8
        assert metadata["strategy"] == "sliding_window"

    def test_max_turns_limit(self, token_counter, sample_messages):
        """Test max turns limiting."""
        strategy = SlidingWindowStrategy({"max_turns": 2})
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Should keep only last 2 turns (4 messages)
        assert len(result) == 4
        assert metadata["turns_included"] == 2
        assert metadata["turns_dropped"] == 2

    def test_token_limit(self, token_counter, sample_messages):
        """Test token budget truncation."""
        strategy = SlidingWindowStrategy({"max_turns": 10})
        result, metadata = strategy.apply(sample_messages, 20, token_counter)

        # Should be truncated to fit token budget
        assert len(result) < len(sample_messages)

    def test_empty_messages(self, token_counter):
        """Test with empty messages."""
        strategy = SlidingWindowStrategy()
        result, metadata = strategy.apply([], 10000, token_counter)

        assert result == []
        assert metadata["turns_included"] == 0

    def test_preserves_message_content(self, token_counter, sample_messages):
        """Test that message content is preserved."""
        strategy = SlidingWindowStrategy({"max_turns": 1})
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Should have last turn
        assert result[-1]["content"] == "Use open() and read() functions."


class TestSlidingWindowWithSummaryStrategy:
    """Tests for SlidingWindowWithSummaryStrategy."""

    def test_default_config(self, token_counter, sample_messages):
        """Test with default configuration."""
        strategy = SlidingWindowWithSummaryStrategy()
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        assert metadata["strategy"] == "sliding_window_with_summary"

    def test_summary_generation(self, token_counter, sample_messages):
        """Test that older messages are summarized."""
        strategy = SlidingWindowWithSummaryStrategy({"max_recent_turns": 1})
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Should have summary + recent messages
        assert metadata["turns_summarized"] == 3
        assert metadata["summary_included"] is True

    def test_custom_summarizer(self, token_counter, sample_messages):
        """Test with custom summarizer function."""
        def custom_summarizer(messages):
            return "Custom summary of conversation"

        strategy = SlidingWindowWithSummaryStrategy({
            "max_recent_turns": 1,
            "summarizer": custom_summarizer,
        })
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Check summary is included
        has_summary = any("Custom summary" in m["content"] for m in result)
        assert has_summary

    def test_no_summary_needed(self, token_counter, sample_messages):
        """Test when all messages fit in recent turns."""
        strategy = SlidingWindowWithSummaryStrategy({"max_recent_turns": 10})
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # No summary needed
        assert metadata["turns_summarized"] == 0


class TestSelectiveStrategy:
    """Tests for SelectiveStrategy."""

    def test_default_config(self, token_counter, sample_messages):
        """Test with default configuration."""
        strategy = SelectiveStrategy()
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        assert metadata["strategy"] == "selective"

    def test_query_relevance(self, token_counter, sample_messages):
        """Test filtering by query relevance."""
        strategy = SelectiveStrategy({
            "query": "Python file read",
            "relevance_threshold": 0.3,
        })
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Should include messages about Python and reading files
        content = " ".join(m["content"] for m in result)
        assert "Python" in content or "file" in content or "read" in content

    def test_custom_similarity_function(self, token_counter, sample_messages):
        """Test with custom similarity function."""
        def always_relevant(query, text):
            return 1.0

        strategy = SelectiveStrategy({
            "query": "anything",
            "similarity_fn": always_relevant,
        })
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # All messages should be included (up to max_turns)
        assert len(result) > 0

    def test_no_query(self, token_counter, sample_messages):
        """Test with no query provided."""
        strategy = SelectiveStrategy({"query": ""})
        result, metadata = strategy.apply(sample_messages, 10000, token_counter)

        # Should return recent messages
        assert len(result) > 0


class TestGetStrategy:
    """Tests for get_strategy factory function."""

    def test_get_sliding_window(self):
        """Test getting sliding window strategy."""
        strategy = get_strategy("sliding_window")
        assert isinstance(strategy, SlidingWindowStrategy)

    def test_get_sliding_window_with_summary(self):
        """Test getting sliding window with summary strategy."""
        strategy = get_strategy("sliding_window_with_summary")
        assert isinstance(strategy, SlidingWindowWithSummaryStrategy)

    def test_get_selective(self):
        """Test getting selective strategy."""
        strategy = get_strategy("selective")
        assert isinstance(strategy, SelectiveStrategy)

    def test_get_with_config(self):
        """Test getting strategy with config."""
        strategy = get_strategy("sliding_window", {"max_turns": 5})
        assert isinstance(strategy, SlidingWindowStrategy)

    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown history strategy"):
            get_strategy("unknown_strategy")
