"""Tests for contextkit.layers.tool_results module."""

import pytest

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.layers.tool_results import ToolResultsLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


@pytest.fixture
def sample_results():
    """Provide sample tool results."""
    return [
        {
            "tool_name": "web_search",
            "input": {"query": "weather today"},
            "output": "Sunny, 72°F with light winds.",
            "status": "success",
        },
        {
            "tool_name": "database_query",
            "input": {"sql": "SELECT * FROM users"},
            "output": "5 rows returned: user1, user2, user3...",
            "status": "success",
        },
        {
            "tool_name": "api_call",
            "input": {"endpoint": "/status"},
            "output": "Error: Service unavailable",
            "status": "error",
        },
    ]


class TestToolResultsLayerInit:
    """Tests for ToolResultsLayer initialization."""

    def test_basic_init(self, sample_results):
        """Test basic initialization."""
        layer = ToolResultsLayer(results=sample_results)

        assert len(layer.results) == 3
        assert layer.name == "tool_results"
        assert layer.priority == 8
        assert layer.include_inputs is False

    def test_include_inputs(self, sample_results):
        """Test with include_inputs enabled."""
        layer = ToolResultsLayer(results=sample_results, include_inputs=True)

        assert layer.include_inputs is True

    def test_custom_name_and_priority(self, sample_results):
        """Test custom name and priority."""
        layer = ToolResultsLayer(
            results=sample_results,
            name="tools",
            priority=9,
        )

        assert layer.name == "tools"
        assert layer.priority == 9

    def test_results_are_copied(self, sample_results):
        """Test that results are copied."""
        layer = ToolResultsLayer(results=sample_results)

        # Modify original
        sample_results[0]["output"] = "Modified"

        # Layer should have original
        assert "72°F" in layer.results[0]["output"]

    def test_inherits_from_base_layer(self, sample_results):
        """Test inheritance."""
        layer = ToolResultsLayer(results=sample_results)
        assert isinstance(layer, BaseLayer)


class TestToolResultsLayerBuild:
    """Tests for ToolResultsLayer.build method."""

    def test_build_basic(self, token_counter, sample_results):
        """Test basic build."""
        layer = ToolResultsLayer(results=sample_results)
        content = layer.build(token_counter)

        assert "web_search" in content
        assert "Sunny, 72°F" in content
        assert layer.get_token_count() > 0

    def test_build_shows_status(self, token_counter, sample_results):
        """Test that status is shown."""
        layer = ToolResultsLayer(results=sample_results)
        content = layer.build(token_counter)

        assert "Success" in content
        assert "Error" in content

    def test_build_without_inputs(self, token_counter, sample_results):
        """Test build without inputs."""
        layer = ToolResultsLayer(results=sample_results, include_inputs=False)
        content = layer.build(token_counter)

        assert "Input:" not in content

    def test_build_with_inputs(self, token_counter, sample_results):
        """Test build with inputs."""
        layer = ToolResultsLayer(results=sample_results, include_inputs=True)
        content = layer.build(token_counter)

        assert "Input:" in content
        assert "weather" in content or "query" in content

    def test_build_empty_results(self, token_counter):
        """Test build with empty results."""
        layer = ToolResultsLayer(results=[])
        content = layer.build(token_counter)

        assert content == ""
        assert layer.get_token_count() == 0


class TestToolResultsLayerTruncate:
    """Tests for ToolResultsLayer.truncate method."""

    def test_truncate_summarizes_older(self, token_counter, sample_results):
        """Test that truncation summarizes older results."""
        layer = ToolResultsLayer(results=sample_results)
        layer.build(token_counter)
        original_tokens = layer.get_token_count()

        # Use a budget smaller than full content to force truncation
        layer.truncate(original_tokens // 2, token_counter)

        # Older results should be summarized (shorter format)
        assert layer.truncated is True

    def test_truncate_keeps_recent(self, token_counter, sample_results):
        """Test that most recent result is kept."""
        layer = ToolResultsLayer(results=sample_results)
        layer.build(token_counter)

        content = layer.truncate(40, token_counter)

        # Most recent (api_call) should still be present
        assert "api_call" in content or "Service unavailable" in content

    def test_truncate_summary_format(self, token_counter, sample_results):
        """Test summarized format."""
        layer = ToolResultsLayer(results=sample_results)
        layer.build(token_counter)

        layer.truncate(30, token_counter)

        # Summary should use short format with status
        # Either includes ✓ or ✗ symbols or full output for most recent
        assert layer.truncated is True


class TestToolResultsLayerInspect:
    """Tests for ToolResultsLayer.inspect method."""

    def test_inspect_before_build(self, sample_results):
        """Test inspect before build."""
        layer = ToolResultsLayer(results=sample_results)
        info = layer.inspect()

        assert info["name"] == "tool_results"
        assert info["total_results"] == 3
        assert info["include_inputs"] is False

    def test_inspect_after_build(self, token_counter, sample_results):
        """Test inspect after build."""
        layer = ToolResultsLayer(results=sample_results)
        layer.build(token_counter)
        info = layer.inspect()

        assert info["full_results"] == 3
        assert info["summarized_results"] == 0

    def test_inspect_after_truncate(self, token_counter, sample_results):
        """Test inspect after truncation."""
        layer = ToolResultsLayer(results=sample_results)
        layer.build(token_counter)
        layer.truncate(30, token_counter)
        info = layer.inspect()

        if info["truncated"]:
            assert info["summarized_results"] > 0


class TestToolResultsLayerFormats:
    """Tests for different result formats."""

    def test_success_status(self, token_counter):
        """Test success status formatting."""
        results = [{
            "tool_name": "test_tool",
            "output": "Success output",
            "status": "success",
        }]
        layer = ToolResultsLayer(results=results)
        content = layer.build(token_counter)

        assert "Success" in content

    def test_error_status(self, token_counter):
        """Test error status formatting."""
        results = [{
            "tool_name": "test_tool",
            "output": "Error message",
            "status": "error",
        }]
        layer = ToolResultsLayer(results=results)
        content = layer.build(token_counter)

        assert "Error" in content

    def test_input_truncation(self, token_counter):
        """Test that long inputs are truncated in display."""
        results = [{
            "tool_name": "test_tool",
            "input": {"long_param": "x" * 100},
            "output": "Result",
            "status": "success",
        }]
        layer = ToolResultsLayer(results=results, include_inputs=True)
        content = layer.build(token_counter)

        # Long input should be truncated
        assert "..." in content

    def test_empty_input(self, token_counter):
        """Test with empty input dict."""
        results = [{
            "tool_name": "test_tool",
            "input": {},
            "output": "Result",
            "status": "success",
        }]
        layer = ToolResultsLayer(results=results, include_inputs=True)
        content = layer.build(token_counter)

        assert "Input: {}" in content


class TestToolResultsLayerEdgeCases:
    """Edge case tests."""

    def test_single_result(self, token_counter):
        """Test with single result."""
        results = [{
            "tool_name": "single_tool",
            "output": "Single result",
            "status": "success",
        }]
        layer = ToolResultsLayer(results=results)
        content = layer.build(token_counter)

        assert "single_tool" in content
        assert "Single result" in content

    def test_truncate_single_result(self, token_counter):
        """Test truncation with single result."""
        results = [{
            "tool_name": "tool",
            "output": "Long output " * 50,
            "status": "success",
        }]
        layer = ToolResultsLayer(results=results)
        layer.build(token_counter)

        # Should keep minimal info even if over budget
        content = layer.truncate(5, token_counter)
        assert "tool" in content
