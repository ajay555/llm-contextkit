"""Tests for contextkit.layers.user_context module."""

import pytest

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.layers.user_context import UserContextLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


class TestUserContextLayerInit:
    """Tests for UserContextLayer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        context = {"name": "Jane", "tier": "Enterprise"}
        layer = UserContextLayer(context=context)

        assert layer.context_data == context
        assert layer.name == "user_context"
        assert layer.priority == 7

    def test_custom_name_and_priority(self):
        """Test custom name and priority."""
        layer = UserContextLayer(
            context={"name": "Jane"},
            name="customer",
            priority=8,
        )
        assert layer.name == "customer"
        assert layer.priority == 8

    def test_context_is_copied(self):
        """Test that context is copied, not referenced."""
        context = {"name": "Jane"}
        layer = UserContextLayer(context=context)

        # Modify original
        context["name"] = "John"

        # Layer should have original value
        assert layer.context_data["name"] == "Jane"

    def test_inherits_from_base_layer(self):
        """Test inheritance."""
        layer = UserContextLayer(context={})
        assert isinstance(layer, BaseLayer)


class TestUserContextLayerBuild:
    """Tests for UserContextLayer.build method."""

    def test_build_single_key(self, token_counter):
        """Test building with single key."""
        layer = UserContextLayer(context={"name": "Jane"})
        content = layer.build(token_counter)

        assert "Name: Jane" in content
        assert layer.get_token_count() > 0

    def test_build_multiple_keys(self, token_counter):
        """Test building with multiple keys."""
        layer = UserContextLayer(
            context={
                "name": "Jane Smith",
                "account_tier": "Enterprise",
                "region": "US-West",
            }
        )
        content = layer.build(token_counter)

        assert "Name: Jane Smith" in content
        assert "Account Tier: Enterprise" in content
        assert "Region: US-West" in content

    def test_build_snake_case_conversion(self, token_counter):
        """Test that snake_case keys are converted to Title Case."""
        layer = UserContextLayer(context={"user_name": "Jane"})
        content = layer.build(token_counter)

        assert "User Name: Jane" in content

    def test_build_empty_context(self, token_counter):
        """Test building with empty context."""
        layer = UserContextLayer(context={})
        content = layer.build(token_counter)

        assert content == ""
        assert layer.get_token_count() == 0


class TestUserContextLayerTruncate:
    """Tests for UserContextLayer.truncate method."""

    def test_truncate_removes_keys(self, token_counter):
        """Test that truncation removes keys."""
        layer = UserContextLayer(
            context={
                "key1": "value1",
                "key2": "value2",
                "key3": "value3",
                "key4": "very long value that takes many tokens " * 10,
            }
        )
        layer.build(token_counter)

        # Truncate to small budget
        layer.truncate(20, token_counter)

        # Should have fewer keys
        assert layer.truncated is True

    def test_truncate_preserves_order(self, token_counter):
        """Test that first keys are preserved."""
        layer = UserContextLayer(
            context={
                "important": "first",
                "less_important": "second " * 20,
            }
        )
        layer.build(token_counter)

        # Truncate to small budget
        content = layer.truncate(15, token_counter)

        # First key should be preserved
        assert "Important: first" in content

    def test_truncate_to_empty(self, token_counter):
        """Test truncation to zero tokens."""
        layer = UserContextLayer(
            context={"name": "Jane " * 50}
        )
        layer.build(token_counter)

        content = layer.truncate(1, token_counter)
        assert content == ""
        assert layer.truncated is True


class TestUserContextLayerInspect:
    """Tests for UserContextLayer.inspect method."""

    def test_inspect_before_build(self):
        """Test inspect before build."""
        layer = UserContextLayer(
            context={"name": "Jane", "tier": "Pro"}
        )
        info = layer.inspect()

        assert info["name"] == "user_context"
        assert info["total_keys"] == 2

    def test_inspect_after_build(self, token_counter):
        """Test inspect after build."""
        layer = UserContextLayer(
            context={"name": "Jane", "tier": "Pro"}
        )
        layer.build(token_counter)
        info = layer.inspect()

        assert info["included_keys"] == 2
        assert info["dropped_keys"] == 0
        assert "name" in info["keys_included"]
        assert "tier" in info["keys_included"]

    def test_inspect_after_truncate(self, token_counter):
        """Test inspect after truncation."""
        layer = UserContextLayer(
            context={
                "key1": "short",
                "key2": "very long " * 50,
            }
        )
        layer.build(token_counter)
        layer.truncate(10, token_counter)
        info = layer.inspect()

        assert info["truncated"] is True
        assert info["dropped_keys"] > 0
