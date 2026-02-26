"""Tests for contextkit.layers.system module."""

import pytest

from llm_contextkit.exceptions import BudgetExceededError
from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.layers.system import SystemLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


class TestSystemLayerInit:
    """Tests for SystemLayer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = SystemLayer(instructions="You are a helpful assistant.")
        assert layer.instructions == "You are a helpful assistant."
        assert layer.name == "system"
        assert layer.priority == 10

    def test_custom_name_and_priority(self):
        """Test custom name and priority."""
        layer = SystemLayer(
            instructions="Test",
            name="my_system",
            priority=15,
        )
        assert layer.name == "my_system"
        assert layer.priority == 15

    def test_with_few_shot_examples(self):
        """Test initialization with few-shot examples."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "Bye", "output": "Goodbye!"},
        ]
        layer = SystemLayer(
            instructions="You are friendly.",
            few_shot_examples=examples,
        )
        assert len(layer.few_shot_examples) == 2

    def test_inherits_from_base_layer(self):
        """Test inheritance."""
        layer = SystemLayer(instructions="Test")
        assert isinstance(layer, BaseLayer)


class TestSystemLayerBuild:
    """Tests for SystemLayer.build method."""

    def test_build_instructions_only(self, token_counter):
        """Test building with instructions only."""
        layer = SystemLayer(instructions="You are a helpful assistant.")
        content = layer.build(token_counter)
        assert content == "You are a helpful assistant."
        assert layer.get_token_count() > 0

    def test_build_with_examples(self, token_counter):
        """Test building with few-shot examples."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
        ]
        layer = SystemLayer(
            instructions="You are friendly.",
            few_shot_examples=examples,
        )
        content = layer.build(token_counter)

        assert "You are friendly." in content
        assert "Example 1:" in content
        assert "User: Hello" in content
        assert "Assistant: Hi there!" in content

    def test_build_updates_token_count(self, token_counter):
        """Test that build updates token count."""
        layer = SystemLayer(instructions="Test instruction")
        assert layer.get_token_count() == 0  # Before build
        layer.build(token_counter)
        assert layer.get_token_count() > 0  # After build

    def test_build_stores_content(self, token_counter):
        """Test that build stores built content."""
        layer = SystemLayer(instructions="Test")
        assert layer.built_content is None
        layer.build(token_counter)
        assert layer.built_content == "Test"


class TestSystemLayerTruncate:
    """Tests for SystemLayer.truncate method."""

    def test_truncate_removes_examples(self, token_counter):
        """Test that truncation removes examples."""
        examples = [
            {"input": "Long example input text " * 10, "output": "Long output " * 10},
            {"input": "Another long example " * 10, "output": "Another output " * 10},
        ]
        layer = SystemLayer(
            instructions="Be helpful.",
            few_shot_examples=examples,
        )
        layer.build(token_counter)
        original_tokens = layer.get_token_count()

        # Truncate to much smaller budget
        layer.truncate(50, token_counter)
        assert layer.get_token_count() < original_tokens
        assert layer.truncated is True

    def test_truncate_keeps_instructions(self, token_counter):
        """Test that truncation always keeps instructions."""
        examples = [
            {"input": "Example", "output": "Output"},
        ]
        layer = SystemLayer(
            instructions="Be helpful.",
            few_shot_examples=examples,
        )
        layer.build(token_counter)

        # Truncate to small budget
        content = layer.truncate(10, token_counter)
        assert "Be helpful." in content

    def test_truncate_raises_if_instructions_too_large(self, token_counter):
        """Test that truncation raises if instructions exceed budget."""
        layer = SystemLayer(
            instructions="This is a very long instruction " * 50,
        )
        layer.build(token_counter)

        with pytest.raises(BudgetExceededError):
            layer.truncate(5, token_counter)  # Way too small

    def test_truncate_partial_examples(self, token_counter):
        """Test truncation keeps as many examples as possible."""
        examples = [
            {"input": "Short", "output": "Short"},
            {"input": "Medium example text", "output": "Medium response"},
            {"input": "Long example text " * 5, "output": "Long response " * 5},
        ]
        layer = SystemLayer(
            instructions="Help.",
            few_shot_examples=examples,
        )
        layer.build(token_counter)

        # Get token count for just instructions + 1 example
        instructions_tokens = token_counter.count("Help.")
        example1_tokens = token_counter.count(
            "Example 1:\nUser: Short\nAssistant: Short"
        )

        # Allow enough for instructions + first example + a bit more
        max_tokens = instructions_tokens + example1_tokens + 30

        content = layer.truncate(max_tokens, token_counter)
        # Should keep at least one example
        assert "Example 1:" in content


class TestSystemLayerInspect:
    """Tests for SystemLayer.inspect method."""

    def test_inspect_before_build(self, token_counter):
        """Test inspect before build."""
        layer = SystemLayer(instructions="Test")
        info = layer.inspect()

        assert info["name"] == "system"
        assert info["priority"] == 10
        assert info["token_count"] == 0
        assert info["truncated"] is False

    def test_inspect_after_build(self, token_counter):
        """Test inspect after build."""
        examples = [{"input": "Hi", "output": "Hello"}]
        layer = SystemLayer(
            instructions="Test",
            few_shot_examples=examples,
        )
        layer.build(token_counter)
        info = layer.inspect()

        assert info["token_count"] > 0
        assert info["total_examples"] == 1
        assert info["included_examples"] == 1
        assert info["dropped_examples"] == 0

    def test_inspect_after_truncate(self, token_counter):
        """Test inspect after truncation."""
        examples = [
            {"input": "Example " * 20, "output": "Output " * 20},
        ]
        layer = SystemLayer(
            instructions="Test",
            few_shot_examples=examples,
        )
        layer.build(token_counter)
        layer.truncate(10, token_counter)
        info = layer.inspect()

        assert info["truncated"] is True
        assert info["dropped_examples"] == 1
