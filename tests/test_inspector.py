"""Tests for contextkit.inspector.debug module."""

import pytest

from llm_contextkit.assembler import ContextAssembler
from llm_contextkit.budget import TokenBudget
from llm_contextkit.inspector.debug import (
    BuildTrace,
    ContextInspector,
    InspectionDiff,
    InspectionReport,
)
from llm_contextkit.layers.system import SystemLayer


@pytest.fixture
def inspector():
    """Provide a ContextInspector with approximate tokenizer."""
    return ContextInspector(tokenizer="approximate")


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great, thanks for asking!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
    ]


class TestContextInspectorInit:
    """Tests for ContextInspector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        inspector = ContextInspector()
        assert inspector.token_counter is not None

    def test_custom_tokenizer(self):
        """Test with custom tokenizer."""
        inspector = ContextInspector(tokenizer="approximate")
        assert inspector.token_counter.name == "approximate"


class TestContextInspectorAnalyze:
    """Tests for ContextInspector.analyze method."""

    def test_analyze_basic(self, inspector, sample_messages):
        """Test basic analysis."""
        report = inspector.analyze(sample_messages)

        assert isinstance(report, InspectionReport)
        assert report.message_count == 5
        assert report.total_tokens > 0

    def test_analyze_token_distribution(self, inspector, sample_messages):
        """Test token distribution calculation."""
        report = inspector.analyze(sample_messages)

        assert "system" in report.token_distribution
        assert "user" in report.token_distribution
        assert "assistant" in report.token_distribution

    def test_analyze_token_percentages(self, inspector, sample_messages):
        """Test token percentage calculation."""
        report = inspector.analyze(sample_messages)

        total_pct = sum(report.token_percentages.values())
        assert abs(total_pct - 100.0) < 0.1  # Should sum to ~100%

    def test_analyze_message_details(self, inspector, sample_messages):
        """Test individual message analysis."""
        report = inspector.analyze(sample_messages)

        assert len(report.messages) == 5
        assert report.messages[0].role == "system"
        assert report.messages[0].index == 0
        assert report.messages[0].token_count > 0

    def test_analyze_empty_messages(self, inspector):
        """Test analysis of empty messages."""
        report = inspector.analyze([])

        assert report.message_count == 0
        assert report.total_tokens == 0

    def test_analyze_estimated_costs(self, inspector, sample_messages):
        """Test cost estimation."""
        report = inspector.analyze(sample_messages)

        assert "gpt-4" in report.estimated_costs
        assert "claude-3-opus" in report.estimated_costs
        assert all(cost >= 0 for cost in report.estimated_costs.values())


class TestContextInspectorWarnings:
    """Tests for warning generation."""

    def test_warning_large_system_prompt(self, inspector):
        """Test warning for large system prompt."""
        messages = [
            {"role": "system", "content": "Instructions " * 500},
            {"role": "user", "content": "Hi"},
        ]
        report = inspector.analyze(messages)

        # Should have warning about large system prompt
        assert any("system prompt" in w.lower() for w in report.warnings)

    def test_warning_empty_content(self, inspector):
        """Test warning for empty message content."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response"},
        ]
        report = inspector.analyze(messages)

        assert any("empty" in w.lower() for w in report.warnings)


class TestContextInspectorDiff:
    """Tests for ContextInspector.diff method."""

    def test_diff_basic(self, inspector, sample_messages):
        """Test basic diff."""
        before = sample_messages[:3]
        after = sample_messages

        diff = inspector.diff(before, after)

        assert isinstance(diff, InspectionDiff)
        assert diff.after_tokens > diff.before_tokens
        assert diff.token_delta > 0

    def test_diff_added_messages(self, inspector):
        """Test detecting added messages."""
        before = [{"role": "user", "content": "Hello"}]
        after = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        diff = inspector.diff(before, after)

        assert len(diff.messages_added) == 1
        assert diff.messages_added[0].role == "assistant"

    def test_diff_removed_messages(self, inspector):
        """Test detecting removed messages."""
        before = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        after = [{"role": "user", "content": "Hello"}]

        diff = inspector.diff(before, after)

        assert len(diff.messages_removed) == 1

    def test_diff_changed_messages(self, inspector):
        """Test detecting changed messages."""
        before = [{"role": "user", "content": "Hello"}]
        after = [{"role": "user", "content": "Hello world"}]

        diff = inspector.diff(before, after)

        assert len(diff.messages_changed) == 1

    def test_diff_no_changes(self, inspector, sample_messages):
        """Test diff with identical messages."""
        diff = inspector.diff(sample_messages, sample_messages.copy())

        assert diff.token_delta == 0
        assert len(diff.messages_added) == 0
        assert len(diff.messages_removed) == 0


class TestContextInspectorTrace:
    """Tests for ContextInspector.trace method."""

    def test_trace_basic(self, inspector):
        """Test basic trace functionality."""
        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        assembler = ContextAssembler(budget=budget)
        assembler.add_layer(SystemLayer(instructions="Be helpful."))

        trace = inspector.trace(assembler)

        assert isinstance(trace, BuildTrace)
        assert len(trace.layers_built) > 0
        assert trace.final_tokens > 0

    def test_trace_no_build(self, inspector):
        """Test trace when no build was done."""
        budget = TokenBudget(total=4096, tokenizer="approximate")
        assembler = ContextAssembler(budget=budget)

        trace = inspector.trace(assembler)

        # Should handle gracefully
        assert isinstance(trace, BuildTrace)


class TestInspectionReportPretty:
    """Tests for InspectionReport.pretty method."""

    def test_pretty_output(self, inspector, sample_messages):
        """Test pretty output generation."""
        report = inspector.analyze(sample_messages)
        output = report.pretty()

        assert isinstance(output, str)
        assert "Context Inspection Report" in output
        assert "Total Tokens" in output
        assert "Token Distribution" in output

    def test_pretty_includes_warnings(self, inspector):
        """Test that pretty output includes warnings."""
        messages = [
            {"role": "system", "content": "Very long " * 1000},
            {"role": "user", "content": "Hi"},
        ]
        report = inspector.analyze(messages)
        output = report.pretty()

        assert "Warnings" in output


class TestInspectionDiffPretty:
    """Tests for InspectionDiff.pretty method."""

    def test_pretty_output(self, inspector, sample_messages):
        """Test diff pretty output."""
        before = sample_messages[:2]
        after = sample_messages

        diff = inspector.diff(before, after)
        output = diff.pretty()

        assert isinstance(output, str)
        assert "Context Diff Report" in output
        assert "Before" in output
        assert "After" in output


class TestBuildTracePretty:
    """Tests for BuildTrace.pretty method."""

    def test_pretty_output(self, inspector):
        """Test trace pretty output."""
        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        assembler = ContextAssembler(budget=budget)
        assembler.add_layer(SystemLayer(instructions="Be helpful."))

        trace = inspector.trace(assembler)
        output = trace.pretty()

        assert isinstance(output, str)
        assert "Build Trace Report" in output
        assert "Build Time" in output
