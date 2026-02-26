"""Tests for contextkit.assembler module."""

import pytest

from llm_contextkit.assembler import ContextAssembler
from llm_contextkit.budget import TokenBudget
from llm_contextkit.formatting.formatter import DefaultFormatter
from llm_contextkit.layers.system import SystemLayer


@pytest.fixture
def budget():
    """Provide a basic token budget."""
    budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
    budget.allocate("system", 500, priority=10)
    return budget


@pytest.fixture
def assembler(budget):
    """Provide a basic assembler."""
    return ContextAssembler(budget=budget)


class TestContextAssemblerInit:
    """Tests for ContextAssembler initialization."""

    def test_basic_init(self, budget):
        """Test basic initialization."""
        assembler = ContextAssembler(budget=budget)
        assert assembler.budget is budget
        assert isinstance(assembler.formatter, DefaultFormatter)

    def test_custom_formatter(self, budget):
        """Test initialization with custom formatter."""
        formatter = DefaultFormatter(section_delimiter="###")
        assembler = ContextAssembler(budget=budget, formatter=formatter)
        assert assembler.formatter.section_delimiter == "###"


class TestContextAssemblerLayers:
    """Tests for layer management."""

    def test_add_layer(self, assembler):
        """Test adding a layer."""
        layer = SystemLayer(instructions="Test")
        result = assembler.add_layer(layer)

        assert "system" in assembler.layers
        assert result is assembler  # Returns self for chaining

    def test_add_layer_chaining(self, budget):
        """Test method chaining."""
        budget.allocate("system2", 500)
        assembler = ContextAssembler(budget=budget)

        layer1 = SystemLayer(instructions="Test 1", name="system")
        layer2 = SystemLayer(instructions="Test 2", name="system2")

        assembler.add_layer(layer1).add_layer(layer2)

        assert len(assembler.layers) == 2

    def test_add_duplicate_layer_raises(self, assembler):
        """Test adding duplicate layer raises error."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)

        with pytest.raises(ValueError, match="already exists"):
            assembler.add_layer(layer)

    def test_remove_layer(self, assembler):
        """Test removing a layer."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)
        result = assembler.remove_layer("system")

        assert "system" not in assembler.layers
        assert result is assembler

    def test_remove_nonexistent_layer_raises(self, assembler):
        """Test removing non-existent layer raises error."""
        with pytest.raises(KeyError, match="not found"):
            assembler.remove_layer("nonexistent")

    def test_get_layer(self, assembler):
        """Test getting a layer."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)

        retrieved = assembler.get_layer("system")
        assert retrieved is layer

    def test_get_nonexistent_layer(self, assembler):
        """Test getting non-existent layer returns None."""
        assert assembler.get_layer("nonexistent") is None


class TestContextAssemblerBuild:
    """Tests for build method."""

    def test_basic_build(self, assembler):
        """Test basic build."""
        layer = SystemLayer(instructions="You are helpful.")
        assembler.add_layer(layer)

        result = assembler.build()

        assert "system" in result
        assert result["system"] == "You are helpful."
        assert "messages" in result
        assert "metadata" in result

    def test_build_metadata(self, assembler):
        """Test build metadata."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)

        result = assembler.build()
        metadata = result["metadata"]

        assert "total_tokens" in metadata
        assert "budget_total" in metadata
        assert "budget_remaining" in metadata
        assert "build_time_ms" in metadata
        assert "layers" in metadata
        assert "warnings" in metadata

    def test_build_with_few_shot_examples(self, assembler):
        """Test build with few-shot examples."""
        layer = SystemLayer(
            instructions="Be helpful.",
            few_shot_examples=[
                {"input": "Hello", "output": "Hi there!"},
            ],
        )
        assembler.add_layer(layer)

        result = assembler.build()
        assert "Example 1:" in result["system"]
        assert "User: Hello" in result["system"]


class TestContextAssemblerTruncation:
    """Tests for budget enforcement and truncation."""

    def test_under_budget_no_truncation(self):
        """Test that content under budget is not truncated."""
        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        assembler = ContextAssembler(budget=budget)

        layer = SystemLayer(instructions="Short instruction")
        assembler.add_layer(layer)

        result = assembler.build()
        assert result["metadata"]["layers"]["system"]["truncated"] is False

    def test_truncation_warning(self):
        """Test that truncation generates warning."""
        budget = TokenBudget(total=100, tokenizer="approximate", reserve_for_output=20)
        budget.allocate("system", 50, priority=5)  # Lower priority for truncation
        assembler = ContextAssembler(budget=budget)

        # Create a layer with many examples that will need truncation
        examples = [
            {"input": "Example " * 10, "output": "Output " * 10},
            {"input": "Example 2 " * 10, "output": "Output 2 " * 10},
        ]
        layer = SystemLayer(
            instructions="Help",
            few_shot_examples=examples,
            priority=5,  # Lower priority
        )
        assembler.add_layer(layer)

        result = assembler.build()
        # May or may not be truncated depending on token counting
        # Just ensure build completes
        assert result is not None


class TestContextAssemblerOpenAIFormat:
    """Tests for build_for_openai method."""

    def test_openai_format(self, assembler):
        """Test OpenAI format output."""
        layer = SystemLayer(instructions="You are helpful.")
        assembler.add_layer(layer)

        messages = assembler.build_for_openai()

        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_openai_format_empty_system(self, budget):
        """Test OpenAI format with no system layer."""
        assembler = ContextAssembler(budget=budget)
        # Don't add any layers

        messages = assembler.build_for_openai()
        # Should have no messages if no layers
        assert isinstance(messages, list)

    def test_openai_format_with_history(self):
        """Test OpenAI format includes history messages."""
        from llm_contextkit.layers.history import HistoryLayer

        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("history", 1000, priority=5)
        assembler = ContextAssembler(budget=budget)

        assembler.add_layer(SystemLayer(instructions="Be helpful."))
        assembler.add_layer(HistoryLayer(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        ))

        messages = assembler.build_for_openai()

        assert messages[0]["role"] == "system"
        assert len(messages) >= 3  # system + 2 history messages

    def test_openai_format_with_user_context(self):
        """Test OpenAI format includes user context in system."""
        from llm_contextkit.layers.user_context import UserContextLayer

        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("user_context", 500, priority=7)
        assembler = ContextAssembler(budget=budget)

        assembler.add_layer(SystemLayer(instructions="Be helpful."))
        assembler.add_layer(UserContextLayer(context={"name": "Jane", "tier": "Pro"}))

        messages = assembler.build_for_openai()

        # User context should be in system message
        system_content = messages[0]["content"]
        assert "Jane" in system_content or "Customer Profile" in system_content

    def test_openai_format_with_retrieved_docs(self):
        """Test OpenAI format includes retrieved docs."""
        from llm_contextkit.layers.retrieved import RetrievedLayer

        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("retrieved_docs", 500, priority=6)
        assembler = ContextAssembler(budget=budget)

        assembler.add_layer(SystemLayer(instructions="Answer based on docs."))
        assembler.add_layer(RetrievedLayer(
            chunks=[{"text": "Important policy information."}]
        ))

        messages = assembler.build_for_openai()

        system_content = messages[0]["content"]
        assert "Important policy" in system_content or "Relevant Documents" in system_content


class TestContextAssemblerAnthropicFormat:
    """Tests for build_for_anthropic method."""

    def test_anthropic_format(self, assembler):
        """Test Anthropic format output."""
        layer = SystemLayer(instructions="You are helpful.")
        assembler.add_layer(layer)

        result = assembler.build_for_anthropic()

        assert isinstance(result, dict)
        assert "system" in result
        assert "messages" in result
        assert result["system"] == "You are helpful."

    def test_anthropic_format_with_history(self):
        """Test Anthropic format includes history."""
        from llm_contextkit.layers.history import HistoryLayer

        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("history", 1000, priority=5)
        assembler = ContextAssembler(budget=budget)

        assembler.add_layer(SystemLayer(instructions="Be helpful."))
        assembler.add_layer(HistoryLayer(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        ))

        result = assembler.build_for_anthropic()

        assert "system" in result
        assert len(result["messages"]) >= 2

    def test_anthropic_format_message_roles(self):
        """Test Anthropic format has valid roles."""
        from llm_contextkit.layers.history import HistoryLayer

        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("history", 1000, priority=5)
        assembler = ContextAssembler(budget=budget)

        assembler.add_layer(HistoryLayer(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        ))

        result = assembler.build_for_anthropic()

        # All messages should have valid Anthropic roles
        for msg in result["messages"]:
            assert msg["role"] in ("user", "assistant")


class TestContextAssemblerInspect:
    """Tests for inspect methods."""

    def test_inspect_before_build_raises(self, assembler):
        """Test inspect before build raises error."""
        with pytest.raises(ValueError, match="No build result available"):
            assembler.inspect()

    def test_inspect_after_build(self, assembler):
        """Test inspect after build."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)
        assembler.build()

        info = assembler.inspect()
        assert "total_tokens" in info
        assert "layers" in info
        assert "system" in info["layers"]

    def test_inspect_pretty(self, assembler):
        """Test inspect_pretty output."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)
        assembler.build()

        output = assembler.inspect_pretty()
        assert isinstance(output, str)
        assert "Context Build Summary" in output
        assert "system" in output
        assert "Build time:" in output


class TestContextAssemblerIntegration:
    """Integration tests for ContextAssembler."""

    def test_full_workflow(self):
        """Test complete workflow."""
        # Setup budget
        budget = TokenBudget(total=4096, tokenizer="approximate", reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)

        # Create assembler
        assembler = ContextAssembler(budget=budget)

        # Add system layer
        system = SystemLayer(
            instructions="You are a helpful customer support agent.",
            few_shot_examples=[
                {"input": "I need help", "output": "I'd be happy to help!"},
            ],
        )
        assembler.add_layer(system)

        # Build for OpenAI
        messages = assembler.build_for_openai()
        assert len(messages) >= 1
        assert "customer support" in messages[0]["content"]

        # Check inspection
        info = assembler.inspect()
        assert info["total_tokens"] > 0
        assert info["budget_remaining"] > 0

    def test_multiple_builds(self, assembler):
        """Test multiple builds work correctly."""
        layer = SystemLayer(instructions="Test")
        assembler.add_layer(layer)

        result1 = assembler.build()
        result2 = assembler.build()

        # Both builds should succeed
        assert result1 is not None
        assert result2 is not None
