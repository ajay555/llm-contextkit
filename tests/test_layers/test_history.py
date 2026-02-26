"""Tests for contextkit.layers.history module."""

import pytest

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.layers.history import HistoryLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


@pytest.fixture
def sample_messages():
    """Provide sample conversation messages."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "It's sunny today."},
    ]


class TestHistoryLayerInit:
    """Tests for HistoryLayer initialization."""

    def test_basic_init(self, sample_messages):
        """Test basic initialization."""
        layer = HistoryLayer(messages=sample_messages)

        assert layer.messages == sample_messages
        assert layer.name == "history"
        assert layer.priority == 5
        assert layer.strategy_name == "sliding_window"

    def test_custom_strategy(self, sample_messages):
        """Test with custom strategy."""
        layer = HistoryLayer(
            messages=sample_messages,
            strategy="sliding_window_with_summary",
            strategy_config={"max_recent_turns": 2},
        )
        assert layer.strategy_name == "sliding_window_with_summary"

    def test_custom_name_and_priority(self, sample_messages):
        """Test custom name and priority."""
        layer = HistoryLayer(
            messages=sample_messages,
            name="conversation",
            priority=6,
        )
        assert layer.name == "conversation"
        assert layer.priority == 6

    def test_messages_are_copied(self, sample_messages):
        """Test that messages are copied."""
        layer = HistoryLayer(messages=sample_messages)

        # Modify original
        sample_messages[0]["content"] = "Modified"

        # Layer should have original
        assert layer.messages[0]["content"] == "Hello!"

    def test_inherits_from_base_layer(self, sample_messages):
        """Test inheritance."""
        layer = HistoryLayer(messages=sample_messages)
        assert isinstance(layer, BaseLayer)


class TestHistoryLayerBuild:
    """Tests for HistoryLayer.build method."""

    def test_build_basic(self, token_counter, sample_messages):
        """Test basic build."""
        layer = HistoryLayer(messages=sample_messages)
        content = layer.build(token_counter)

        assert "User: Hello!" in content
        assert "Assistant: Hi there!" in content
        assert layer.get_token_count() > 0

    def test_build_empty_messages(self, token_counter):
        """Test build with empty messages."""
        layer = HistoryLayer(messages=[])
        content = layer.build(token_counter)

        assert content == ""
        assert layer.get_token_count() == 0

    def test_build_with_strategy(self, token_counter, sample_messages):
        """Test build with strategy applied."""
        layer = HistoryLayer(
            messages=sample_messages,
            strategy="sliding_window",
            strategy_config={"max_turns": 1},
        )
        content = layer.build(token_counter)

        # Should only have last turn
        assert "What's the weather?" in content
        assert "Hello!" not in content

    def test_processed_messages(self, token_counter, sample_messages):
        """Test processed_messages property."""
        layer = HistoryLayer(
            messages=sample_messages,
            strategy="sliding_window",
            strategy_config={"max_turns": 1},
        )
        layer.build(token_counter)

        processed = layer.processed_messages
        assert len(processed) == 2  # 1 turn = 2 messages


class TestHistoryLayerTruncate:
    """Tests for HistoryLayer.truncate method."""

    def test_truncate_by_tokens(self, token_counter, sample_messages):
        """Test truncation by token budget."""
        layer = HistoryLayer(messages=sample_messages)
        layer.build(token_counter)

        layer.truncate(20, token_counter)

        # Should be truncated
        assert layer.get_token_count() <= 20

    def test_truncate_sets_flag(self, token_counter, sample_messages):
        """Test that truncate sets truncated flag."""
        layer = HistoryLayer(messages=sample_messages)
        layer.build(token_counter)

        original_count = layer.get_token_count()
        layer.truncate(10, token_counter)

        if layer.get_token_count() < original_count:
            assert layer.truncated is True


class TestHistoryLayerApiFormat:
    """Tests for HistoryLayer.get_messages_for_api method."""

    def test_api_format(self, token_counter, sample_messages):
        """Test getting messages in API format."""
        layer = HistoryLayer(messages=sample_messages)
        layer.build(token_counter)

        api_messages = layer.get_messages_for_api()

        assert len(api_messages) == 4
        assert all("role" in m and "content" in m for m in api_messages)


class TestHistoryLayerInspect:
    """Tests for HistoryLayer.inspect method."""

    def test_inspect_before_build(self, sample_messages):
        """Test inspect before build."""
        layer = HistoryLayer(messages=sample_messages)
        info = layer.inspect()

        assert info["name"] == "history"
        assert info["strategy"] == "sliding_window"
        assert info["total_messages"] == 4

    def test_inspect_after_build(self, token_counter, sample_messages):
        """Test inspect after build."""
        layer = HistoryLayer(messages=sample_messages)
        layer.build(token_counter)
        info = layer.inspect()

        assert info["processed_messages"] == 4
        assert "strategy_metadata" in info

    def test_inspect_with_truncation(self, token_counter, sample_messages):
        """Test inspect after truncation."""
        layer = HistoryLayer(
            messages=sample_messages,
            strategy="sliding_window",
            strategy_config={"max_turns": 1},
        )
        layer.build(token_counter)
        info = layer.inspect()

        assert info["processed_messages"] == 2
        assert info["strategy_metadata"]["turns_included"] == 1


class TestHistoryLayerStrategies:
    """Integration tests for different strategies."""

    def test_sliding_window(self, token_counter):
        """Test sliding window strategy integration."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ] + [
            {"role": "assistant", "content": f"Response {i}"}
            for i in range(10)
        ]
        # Interleave them properly
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        layer = HistoryLayer(
            messages=messages,
            strategy="sliding_window",
            strategy_config={"max_turns": 3},
        )
        layer.build(token_counter)

        # Should only have 3 turns = 6 messages
        assert len(layer.processed_messages) == 6

    def test_selective_strategy(self, token_counter):
        """Test selective strategy integration."""
        messages = [
            {"role": "user", "content": "Tell me about Python programming"},
            {"role": "assistant", "content": "Python is a great language."},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "It's sunny."},
            {"role": "user", "content": "How do I write Python code?"},
            {"role": "assistant", "content": "Start with print('Hello')."},
        ]

        layer = HistoryLayer(
            messages=messages,
            strategy="selective",
            strategy_config={
                "query": "Python programming",
                "relevance_threshold": 0.3,
            },
        )
        layer.build(token_counter)

        # Should include Python-related messages
        content = layer.built_content
        assert "Python" in content
